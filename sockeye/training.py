# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Code for training
"""
import glob
import logging
import multiprocessing as mp
import os
import pickle
import random
import shutil
import time
from functools import reduce
from typing import AnyStr, Any, Dict, List, Optional, Tuple, Union

import mxnet as mx
import numpy as np
from math import sqrt
from . import callback

from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
from . import lr_scheduler
from . import model
from . import utils
from . import vocab
from .optimizers import BatchState, CheckpointState, SockeyeOptimizer, OptimizerConfig

logger = logging.getLogger(__name__)

class _TrainingState:
    """
    Stores the state of the training process. These are the variables that will
    be stored to disk when resuming training.
    """

    def __init__(self,
                 num_not_improved,
                 epoch,
                 checkpoint,
                 updates,
                 samples):
        self.num_not_improved = num_not_improved
        self.epoch = epoch
        self.checkpoint = checkpoint
        self.updates = updates
        self.samples = samples


class BaseTrainingModel(model.SockeyeModel):
    """
    Defines an Encoder/Decoder model (with attention).
    RNN configuration (number of hidden units, number of layers, cell type)
    is shared between encoder & decoder.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param train_iter: The iterator over the training data.
    :param fused: If True fused RNN cells will be used (should be slightly more efficient, but is only available
            on GPUs).
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param lr_scheduler: The scheduler that lowers the learning rate during training.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 state_names: Optional[List[str]] = None,
                 grad_req: Optional[str] = 'write') -> None:
        super().__init__(config)
        self.context = context
        self.lr_scheduler = lr_scheduler
        self.bucketing = bucketing
        self.state_names = state_names if state_names is not None else []
        self.grad_req = grad_req

        self._build_model_components(self.config.max_seq_len, fused)
        self.module = self._build_module(train_iter, self.config.max_seq_len)
        self.training_monitor = None

    def _build_module(self,
                      train_iter: data_io.ParallelBucketSentenceIter,
                      max_seq_len: int):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        default_bucket_key = train_iter.default_bucket_key
        model_symbols = self._define_symbols()
        in_out_names = self._define_names(train_iter)
        sym_gen = self._create_computation_graph(model_symbols, in_out_names)

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          state_names=self.state_names,
                                          logger=logger,
                                          default_bucket_key=train_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to max_seq_len=%s", max_seq_len)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 state_names=self.state_names,
                                 logger=logger,
                                 context=self.context)

    def _define_symbols(self):
        raise NotImplementedError()

    def _define_names(self, train_iter):
        raise NotImplementedError()

    def _create_computation_graph(self, model_symbols, in_out_names):
        raise NotImplementedError()

    def _get_shapes(self, train_iter: data_io.ParallelBucketSentenceIter):
        raise NotImplementedError()

    def _get_data_batch(self, data_iter):
        raise NotImplementedError()

    def _prepare_metric_update(self, batch):
        raise NotImplementedError()

    def _compute_gradients(self, batch: mx.io.DataBatch, train_state):
        raise NotImplementedError()

    def _clear_gradients(self):
        for grad in self.module._curr_module._exec_group.grad_arrays:
            grad[0][:] = 0

    @staticmethod
    def _create_eval_metric(metric_names: List[AnyStr]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = []
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        for metric_name in metric_names:
            if metric_name == C.ACCURACY:
                metrics.append(utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            elif metric_name == C.sBLEU:
                metrics.append(utils.SentenceBleu(output_names=[C.GEN_WORDS_OUTPUT_NAME]))
            elif metric_name == C.PERPLEXITY:
                metrics.append(mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            else:
                raise ValueError("unknown metric name")
        return mx.metric.create(metrics)

    def _set_states(self, state_names, values):
        states = self.module.get_states(merge_multi_context=False)
        for state_name, state, value in zip(state_names, states, values):
            for state_per_dev in state:
                state_per_dev[:] = value

    def fit(self,
            train_iter: data_io.ParallelBucketSentenceIter,
            val_iter: data_io.ParallelBucketSentenceIter,
            output_folder: str,
            max_params_files_to_keep: int,
            metrics: List[AnyStr],
            initializer: mx.initializer.Initializer,
            max_updates: int,
            checkpoint_frequency: int,
            optimizer: str,
            optimizer_params: dict,
            optimized_metric: str = "perplexity",
            max_num_not_improved: int = 3,
            min_num_epochs: Optional[int] = None,
            monitor_bleu: int = 0,
            use_tensorboard: bool = False,
            mxmonitor_pattern: Optional[str] = None,
            mxmonitor_stat_func: Optional[str] = None):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator.
        :param val_iter: The validation data iterator.
        :param output_folder: The folder in which all model artifacts will be stored in (parameters, checkpoints, etc.).
        :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
        :param metrics: The metrics that will be evaluated during training.
        :param initializer: The parameter initializer.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing in number of updates.
        :param optimizer: The MXNet optimizer that will update the parameters.
        :param optimizer_params: The parameters for the optimizer.
        :param optimized_metric: The metric that is tracked for early stopping.
        :param max_num_not_improved: Stop training if the optimized_metric does not improve for this many checkpoints.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        :param monitor_bleu: Monitor BLEU during training (0: off, >=0: the number of sentences to decode for BLEU
               evaluation, -1: decode the full validation set.).
        :param use_tensorboard: If True write tensorboard compatible logs for monitoring training and
               validation metrics.
        :param mxmonitor_pattern: Optional pattern to match to monitor weights/gradients/outputs
               with MXNet's monitor. Default is None which means no monitoring.
        :param mxmonitor_stat_func: Choice of statistics function to run on monitored weights/gradients/outputs
               when using MXNEt's monitor.
        :return: Best score on validation data observed during training.
        """
        self.save_version(output_folder)
        self.save_config(output_folder)

        data_shapes, label_shapes = self._get_shapes(train_iter)

        self.module.bind(data_shapes=data_shapes, label_shapes=label_shapes,
                         for_training=True, force_rebind=True, grad_req=self.grad_req)
        self.module.symbol.save(os.path.join(output_folder, C.SYMBOL_NAME))

        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False)

        self.module.init_optimizer(kvstore='device', optimizer=optimizer, optimizer_params=optimizer_params)

        cp_decoder = checkpoint_decoder.CheckpointDecoder(self.context[-1],
                                                          self.config.config_data.validation_source,
                                                          self.config.config_data.validation_target,
                                                          output_folder, self.config.max_seq_len,
                                                          limit=monitor_bleu) \
            if monitor_bleu else None

        logger.info("Training started.")
        self.training_monitor = callback.TrainingMonitor(train_iter.batch_size, output_folder,
                                                         optimized_metric=optimized_metric,
                                                         use_tensorboard=use_tensorboard,
                                                         checkpoint_decoder=cp_decoder)

        monitor = None
        if mxmonitor_pattern is not None:
            monitor = mx.monitor.Monitor(interval=C.MEASURE_SPEED_EVERY,
                                         stat_func=C.MONITOR_STAT_FUNCS.get(mxmonitor_stat_func),
                                         pattern=mxmonitor_pattern,
                                         sort=True)
            self.module.install_monitor(monitor)
            logger.info("Installed MXNet monitor; pattern='%s'; statistics_func='%s'",
                        mxmonitor_pattern, mxmonitor_stat_func)

        self._fit(train_iter, val_iter, output_folder,
                  max_params_files_to_keep,
                  metrics=metrics,
                  max_updates=max_updates,
                  checkpoint_frequency=checkpoint_frequency,
                  max_num_not_improved=max_num_not_improved,
                  min_num_epochs=min_num_epochs,
                  mxmonitor=monitor)

        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.training_monitor.get_best_checkpoint(),
                    self.training_monitor.optimized_metric,
                    self.training_monitor.get_best_validation_score())
        return self.training_monitor.get_best_validation_score()

    def _fit(self,
             train_iter: data_io.ParallelBucketSentenceIter,
             val_iter: data_io.ParallelBucketSentenceIter,
             output_folder: str,
             max_params_files_to_keep: int,
             metrics: List[AnyStr],
             max_updates: int,
             checkpoint_frequency: int,
             max_num_not_improved: int,
             min_num_epochs: Optional[int] = None,
             mxmonitor: Optional[mx.monitor.Monitor] = None):
        """
        Internal fit method. Runtime determined by early stopping.

        :param train_iter: Training data iterator.
        :param val_iter: Validation data iterator.
        :param output_folder: Model output folder.
        :params max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
        :param metrics: List of metric names to track on training and validation data.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing.
        :param max_num_not_improved: Maximum number of checkpoints until fitting is stopped if model does not improve.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        :param mxmonitor: Optional MXNet monitor instance.
        """
        metric_train = self._create_eval_metric(metrics)
        metric_val = self._create_eval_metric(metrics)

        tic = time.time()

        training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(training_state_dir):
            train_state = self.load_checkpoint(training_state_dir, train_iter)
        else:
            train_state = _TrainingState(
                num_not_improved=0,
                epoch=0,
                checkpoint=0,
                updates=0,
                samples=0
            )

        next_data_batch = self._get_data_batch(train_iter)

        while max_updates == -1 or train_state.updates < max_updates:
            if not train_iter.iter_next():
                train_state.epoch += 1
                train_iter.reset()

            # process batch
            batch = next_data_batch

            if mxmonitor is not None:
                mxmonitor.tic()

            # compute the gradients
            self._compute_gradients(batch, train_state)

            # update the model parameters usinge the gradients
            self.module.update()

            if mxmonitor is not None:
                results = mxmonitor.toc()
                if results:
                    for _, k, v in results:
                        logger.info('Monitor: Batch [{:d}] {:s} {:s}'.format(train_state.updates, k, v))

            # clear the gradients if necessary
            if self.grad_req == 'add':
                self._clear_gradients()

            if train_iter.iter_next():
                # pre-fetch next batch
                next_data_batch = self._get_data_batch(train_iter)
                self.module.prepare(next_data_batch)

            batch = self._prepare_metric_update(batch)
            self.module.update_metric(metric_train, batch.label)

            self.training_monitor.batch_end_callback(train_state.epoch, train_state.updates, metric_train)
            train_state.updates += 1
            train_state.samples += train_iter.batch_size

            if train_state.updates > 0 and train_state.updates % checkpoint_frequency == 0:
                train_state.checkpoint += 1
                self._save_params(output_folder, train_state.checkpoint)
                cleanup_params_files(output_folder, max_params_files_to_keep,
                                     train_state.checkpoint, self.training_monitor.get_best_checkpoint())
                self.training_monitor.checkpoint_callback(train_state.checkpoint, metric_train)

                toc = time.time()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f",
                            train_state.checkpoint, train_state.updates, train_state.epoch,
                            train_state.samples, (toc - tic))
                tic = time.time()

                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', train_state.checkpoint, name, val)
                metric_train.reset()

                # evaluation on validation set
                has_improved, best_checkpoint = self._evaluate(train_state, val_iter, metric_val)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.new_evaluation_result(has_improved)

                if has_improved:
                    best_path = os.path.join(output_folder, C.PARAMS_BEST_NAME)
                    if os.path.lexists(best_path):
                        os.remove(best_path)
                    actual_best_fname = C.PARAMS_NAME % best_checkpoint
                    os.symlink(actual_best_fname, best_path)
                    train_state.num_not_improved = 0
                else:
                    train_state.num_not_improved += 1
                    logger.info("Model has not improved for %d checkpoints", train_state.num_not_improved)

                if train_state.num_not_improved >= max_num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, train_state.num_not_improved)
                    stop_fit = True

                    if min_num_epochs is not None and train_state.epoch < min_num_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_num_epochs,
                                    train_state.epoch)
                        stop_fit = False

                    if stop_fit:
                        logger.info("Stopping fit")
                        self.training_monitor.stop_fit_callback()
                        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
                        if os.path.exists(final_training_state_dirname):
                            shutil.rmtree(final_training_state_dirname)
                        break

                self._checkpoint(train_state, output_folder, train_iter)
        cleanup_params_files(output_folder, max_params_files_to_keep,
                             train_state.checkpoint, self.training_monitor.get_best_checkpoint())

    def _save_params(self, output_folder: str, checkpoint: int):
        """
        Saves the parameters to disk.
        """
        arg_params, aux_params = self.module.get_params()  # sync aux params across devices
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        params_base_fname = C.PARAMS_NAME % checkpoint
        self.save_params_to_file(os.path.join(output_folder, params_base_fname))

    def _toggle_states(self, is_train=False):
        raise NotImplementedError()

    def _evaluate(self, training_state, val_iter, val_metric):
        """
        Computes val_metric on val_iter. Returns whether model improved or not.
        """
        val_iter.reset()
        val_metric.reset()

        # set the internal states for making predictions
        self._toggle_states(is_train=False)

        while val_iter.iter_next():
            eval_batch = self._get_data_batch(val_iter)
            self.module.forward(eval_batch, is_train=False)

            eval_batch = self._prepare_metric_update(eval_batch)
            self.module.update_metric(val_metric, eval_batch.label)

        # restore the states back
        self._toggle_states(is_train=True)

        for name, val in val_metric.get_name_value():
            logger.info('Checkpoint [%d]\tValidation-%s=%f', training_state.checkpoint, name, val)

        return self.training_monitor.eval_end_callback(training_state.checkpoint, val_metric)

    def _checkpoint(self, training_state: _TrainingState, output_folder: str,
                    train_iter: data_io.ParallelBucketSentenceIter):
        """
        Saves checkpoint. Note that the parameters are saved in _save_params.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)
        # Link current parameter file
        params_base_fname = C.PARAMS_NAME % training_state.checkpoint
        os.symlink(os.path.join("..", params_base_fname),
                   os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME))

        # Optimizer state (from mxnet)
        opt_state_fname = os.path.join(training_state_dirname, C.MODULE_OPT_STATE_NAME)
        if self.bucketing:
            # This is a bit hacky, as BucketingModule does not provide a
            # save_optimizer_states call. We take the current active module and
            # save its state. This should work, as the individual modules in
            # BucketingModule share the same optimizer through
            # borrow_optimizer.
            self.module._curr_module.save_optimizer_states(opt_state_fname)
        else:
            self.module.save_optimizer_states(opt_state_fname)

        # State of the bucket iterator
        train_iter.save_state(os.path.join(training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp)  # Yes, one uses _, the other does not

        # Monitor state, in order to get the full information about the metrics
        self.training_monitor.save_state(os.path.join(training_state_dirname, C.MONITOR_STATE_NAME))

        # Our own state
        self.save_state(training_state, os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # The lr scheduler
        with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
            pickle.dump(self.lr_scheduler, fp)

        # We are now finished with writing. Rename the temporary directory to
        # the actual directory
        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(final_training_state_dirname):
            os.rename(final_training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, final_training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            shutil.rmtree(delete_training_state_dirname)

    def save_state(self, training_state: _TrainingState, fname: str):
        """
        Saves the state (of the TrainingModel class) to disk.

        :param fname: file name to save the state to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(training_state, fp)

    def load_state(self, fname: str) -> _TrainingState:
        """
        Loads the training state (of the TrainingModel class) from disk.

        :param fname: file name to load the state from.
        """
        training_state = None
        with open(fname, "rb") as fp:
            training_state = pickle.load(fp)
        return training_state

    def load_checkpoint(self, directory: str, train_iter: data_io.ParallelBucketSentenceIter) -> _TrainingState:
        """
        Loads the full training state from disk. This includes optimizer,
        random number generators and everything needed.  Note that params
        should have been loaded already by the initializer.

        :param directory: directory where the state has been saved.
        :param train_iter: training data iterator.
        """

        # Optimzer state (from mxnet)
        opt_state_fname = os.path.join(directory, C.MODULE_OPT_STATE_NAME)
        if self.bucketing:
            # Same hacky solution as for saving the state
            self.module._curr_module.load_optimizer_states(opt_state_fname)
        else:
            self.module.load_optimizer_states(opt_state_fname)

        # State of the bucket iterator
        train_iter.load_state(os.path.join(directory, C.BUCKET_ITER_STATE_NAME))

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(directory, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            np.random.set_state(pickle.load(fp))

        # Monitor state, in order to get the full information about the metrics
        self.training_monitor.load_state(os.path.join(directory, C.MONITOR_STATE_NAME))

        # And our own state
        return self.load_state(os.path.join(directory, C.TRAINING_STATE_NAME))


class MLETrainingModel(BaseTrainingModel):
    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 state_names: Optional[List[str]] = None,
                 grad_req: Optional[str]='write') -> None:
        super().__init__(config, context, train_iter, fused, bucketing, lr_scheduler, state_names, grad_req)

    def _define_symbols(self) -> List[mx.sym.Symbol]:
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        model_loss = loss.get_loss(self.config.config_loss)

        return [source, source_length, target, labels, model_loss]

    def _define_names(self, train_iter) -> Tuple[List['str'], List['str']]:
        data_names = [x[0] for x in train_iter.provide_data]
        label_names = [x[0] for x in train_iter.provide_label]

        return (data_names, label_names)

    def _create_computation_graph(self, model_symbols, in_out_names) -> Tuple[mx.sym.Symbol, List['str'], List['str']]:
        source, source_length, target, labels, model_loss = model_symbols
        data_names, label_names = in_out_names

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            if self.config.config_decoder.scheduled_sampling_type is not None:
                # scheduled sampling
                probs, threshold = self.decoder.decode_iter(source_encoded, source_encoded_seq_len, source_encoded_length,
                                                            target, target_seq_len, source_lexicon)

                cross_entropy = mx.sym.one_hot(indices=mx.sym.cast(data=labels, dtype='int32'),
                                               depth=self.config.vocab_target_size)

                # zero out pad symbols (0)
                cross_entropy = mx.sym.where(labels,
                                             cross_entropy, mx.sym.zeros((0, self.config.vocab_target_size)))

                # compute cross_entropy
                cross_entropy = - mx.sym.log(data=probs + 1e-10) * cross_entropy
                cross_entropy = mx.sym.sum(data=cross_entropy, axis=1)

                cross_entropy = mx.sym.MakeLoss(cross_entropy, name=C.CROSS_ENTROPY)

                probs = mx.sym.BlockGrad(probs, name=C.SOFTMAX_NAME)
                threshold = mx.sym.BlockGrad(threshold)

                outputs = [cross_entropy, probs, threshold]
            else:
                logits = self.decoder.decode(source_encoded, source_encoded_seq_len, source_encoded_length,
                                            target, target_seq_len, source_lexicon)

                outputs = model_loss.get_loss(logits, labels)

            return mx.sym.Group(outputs), data_names, label_names

        return sym_gen

    def _get_shapes(self, train_iter) -> Tuple[mx.io.DataDesc, mx.io.DataDesc]:
        data_shapes = train_iter.provide_data
        label_shapes = train_iter.provide_label

        return (data_shapes, label_shapes)

    def _get_data_batch(self, data_iter) -> mx.io.DataBatch:
        next_data_batch = data_iter.next()

        return next_data_batch

    def _prepare_metric_update(self, batch) -> mx.io.DataBatch:
        return batch

    def _compute_gradients(self, batch: mx.io.DataBatch, train_state):
        if self.state_names is not None:
            # XXX update the internal states which are used for the scheduled sampling
            self._set_states(self.state_names, [eval('train_state.updates')])

        self.module.forward_backward(batch)

    def _toggle_states(self, is_train=False):
        # if prediction is requested by setting 'is_train = False', the model's internal states are all zero (initial values).
        self._set_states(self.state_names, [int(is_train)])


class MRTrainingModel(BaseTrainingModel):
    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 state_names,
                 grad_req) -> None:
        self.mrt_metric = config.config_decoder.mrt_metric
        self.mrt_entropy_reg = config.config_decoder.mrt_entropy_reg
        self.mrt_num_samples = config.config_decoder.mrt_num_samples
        self.mrt_sup_grad_scale = config.config_decoder.mrt_sup_grad_scale
        self.mrt_max_target_seq_ratio = config.config_decoder.mrt_max_target_len_ratio

        super().__init__(config, context, train_iter, fused, bucketing, lr_scheduler, state_names, grad_req)

    def _define_symbols(self) -> List[mx.sym.Symbol]:
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.Variable(C.TARGET_LABEL_NAME)
        baseline = mx.sym.Variable('baseline')
        is_sample = mx.sym.Variable('is_sample', shape=(1,), dtype='int32')

        return [source, source_length, target, labels, baseline, is_sample]

    def _define_names(self, train_iter) -> Tuple[List['str'], List['str']]:
        batch_size = train_iter.batch_size
        data_shapes = train_iter.provide_data + [mx.io.DataDesc(name='baseline', shape=(batch_size,), layout=C.BATCH_MAJOR)]
        label_shapes = train_iter.provide_label

        data_names = [x[0] for x in data_shapes]
        label_names = [x[0] for x in label_shapes]

        return data_names, label_names

    def _create_computation_graph(self, model_symbols, in_out_names) -> Tuple[mx.sym.Symbol, List['str'], List['str']]:
        source, source_length, target, labels, baseline, is_sample = model_symbols
        data_names, label_names = in_out_names

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            outputs = self.decoder.decode_mrt(source_encoded, source_seq_len, source_length,
                                              labels, target_seq_len, self.mrt_max_target_seq_ratio, is_sample,
                                              self.mrt_entropy_reg, self.mrt_sup_grad_scale, source_lexicon)
            sampled_words, sample_probs = outputs

            mrt_loss = mx.sym.Custom(data=sampled_words, label=labels,
                                     baseline=baseline, is_sampled=is_sample,
                                     metric=self.mrt_metric,
                                     ignore_ids=[C.PAD_ID],
                                     eos_id=C.EOS_ID,
                                     name=C.TARGET_NAME, op_type='Risk')

            softmax_output = mx.sym.reshape(sample_probs,
                                            shape=(-1, self.config.vocab_target_size))

            losses = mx.sym.Group([mrt_loss,
                                   mx.sym.BlockGrad(sampled_words, name=C.GEN_WORDS_NAME),
                                   mx.sym.BlockGrad(softmax_output, name=C.SOFTMAX_NAME),
                                   mx.sym.BlockGrad(target)])


            return losses, data_names, label_names

        return sym_gen

    def _get_shapes(self, train_iter) -> Tuple[mx.io.DataDesc, mx.io.DataDesc]:
        data_shapes = self._create_mrt_data_shapes(train_iter.provide_data, train_iter.batch_size)
        label_shapes = train_iter.provide_label

        return (data_shapes, label_shapes)

    def _get_data_batch(self, data_iter) -> mx.io.DataBatch:
        next_data_batch = data_iter.next()
        next_data_batch = self._create_mrt_batch(next_data_batch)

        return next_data_batch

    def _prepare_metric_update(self, batch) -> mx.io.DataBatch:
        batch.label = self._expand_label_seq(batch.label, max_target_seq_len_ratio=self.mrt_max_target_seq_ratio)

        return batch

    def _compute_gradients(self, batch: mx.io.DataBatch, train_state):
        self.module.forward_backward(batch)

        # XXX run multiple forward passes to collect the average reward
        self._set_states(self.state_names, [1]) # is_sample = 1

        if self.mrt_num_samples > 0:
            baseline = batch.data[-1]
            for i in range(self.mrt_num_samples):
                self.module.forward(batch, is_train=False)
                risk_output = self.module.get_outputs()[0].as_in_context(baseline.context)
                baseline[:] += risk_output
            baseline[:] += 1
            baseline[:] /= (self.mrt_num_samples + 1)

        self.module.forward_backward(batch)

        self._set_states(self.state_names, [0]) # is_sample = 0

    def _toggle_states(self, is_train=False):
        # if prediction is requested by setting 'is_train = False', the model will generate sample sequences.
        # That is, the model generates a sequence of words following word distribution (, or the policy).

        self._set_states(self.state_names, [1-int(is_train)])


    @staticmethod
    def _create_mrt_data_shapes(data_shapes, batch_size):
        new_data_shapes = data_shapes + \
            [mx.io.DataDesc(name='baseline', shape=(batch_size,),
                            layout=C.BATCH_MAJOR)]

        return new_data_shapes

    @staticmethod
    def _create_mrt_batch(data_batch: mx.io.DataBatch):
        batch_size = data_batch.label[0].shape[0]

        data_batch.data = data_batch.data + [mx.nd.zeros((batch_size,))]
        data_batch.provide_data = data_batch.provide_data + \
            [mx.io.DataDesc(name='baseline', shape=(batch_size,),
                            layout=C.BATCH_MAJOR)]

        return data_batch

    @staticmethod
    def _expand_label_seq(labels: List[mx.nd.NDArray],
                          max_target_seq_len_ratio: float=1.5):

        for label_idx in range(len(labels)):
            batch_size, target_seq_len = labels[label_idx].shape
            diff_target_seq_len = int(target_seq_len*max_target_seq_len_ratio) - target_seq_len
            labels[label_idx] = mx.nd.concat(*[labels[label_idx],
                                               mx.nd.zeros((batch_size, diff_target_seq_len))],
                                             dim=1)

        return labels


def cleanup_params_files(output_folder: str, max_to_keep: int, checkpoint: int, best_checkpoint: int):
    """
    Cleanup the params files in the output folder.

    :param output_folder: folder where param files are created.
    :param max_to_keep: maximum number of files to keep, negative to keep all.
    :param checkpoint: current checkpoint (i.e. index of last params file created).
    :param best_checkpoint: best checkpoint, we will not delete its params.
    """
    if max_to_keep <= 0:  # We assume we do not want to delete all params
        return
    existing_files = glob.glob(os.path.join(output_folder, C.PARAMS_PREFIX + "*"))
    params_name_with_dir = os.path.join(output_folder, C.PARAMS_NAME)
    for n in range(1, max(1, checkpoint - max_to_keep + 1)):
        if n != best_checkpoint:
            param_fname_n = params_name_with_dir % n
            if param_fname_n in existing_files:
                os.remove(param_fname_n)


class TrainingModel(model.SockeyeModel):
    """
    TrainingModel is a SockeyeModel that fully unrolls over source and target sequences.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU).
    :param output_dir: Directory where this model is stored.
    :param provide_data: List of input data descriptions.
    :param provide_label: List of label descriptions.
    :param default_bucket_key: Default bucket key.
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param gradient_compression_params: Optional dictionary of gradient compression parameters.
    :param fixed_param_names: Optional list of params to fix during training (i.e. their values will not be trained).
    """

    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 output_dir: str,
                 provide_data: List[mx.io.DataDesc],
                 provide_label: List[mx.io.DataDesc],
                 default_bucket_key: Tuple[int, int],
                 bucketing: bool,
                 gradient_compression_params: Optional[Dict[str, Any]] = None,
                 fixed_param_names: Optional[List[str]] = None) -> None:
        super().__init__(config)
        self.context = context
        self.output_dir = output_dir
        self.fixed_param_names = fixed_param_names
        self._bucketing = bucketing
        self._gradient_compression_params = gradient_compression_params
        self._initialize(provide_data, provide_label, default_bucket_key)
        self._monitor = None  # type: Optional[mx.monitor.Monitor]

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    provide_label: List[mx.io.DataDesc],
                    default_bucket_key: Tuple[int, int]):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        self.model_loss = loss.get_loss(self.config.config_loss)

        data_names = [C.SOURCE_NAME, C.TARGET_NAME]
        label_names = [C.TARGET_LABEL_NAME]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))
        provide_label_names = [d[0] for d in provide_label]
        utils.check_condition(provide_label_names == label_names,
                              "incompatible provide_label: %s, names should be %s" % (provide_label_names, label_names))

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, decoder_depth)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(target_decoded)

            loss_output = self.model_loss.get_loss(logits, labels)

            return mx.sym.Group(loss_output), data_names, label_names

        if self.config.lhuc:
            arguments = sym_gen(default_bucket_key)[0].list_arguments()
            fixed_param_names = [a for a in arguments if not a.endswith(C.LHUC_NAME)]
        else:
            fixed_param_names = self.fixed_param_names

        if self._bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", default_bucket_key)
            self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=default_bucket_key,
                                                 context=self.context,
                                                 compression_params=self._gradient_compression_params,
                                                 fixed_param_names=fixed_param_names)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_target)
            symbol, _, __ = sym_gen(default_bucket_key)
            self.module = mx.mod.Module(symbol=symbol,
                                        data_names=data_names,
                                        label_names=label_names,
                                        logger=logger,
                                        context=self.context,
                                        compression_params=self._gradient_compression_params,
                                        fixed_param_names=fixed_param_names)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=provide_label,
                         for_training=True,
                         force_rebind=True,
                         grad_req='write')

        self.module.symbol.save(os.path.join(self.output_dir, C.SYMBOL_NAME))

        self.save_version(self.output_dir)
        self.save_config(self.output_dir)

    def run_forward_backward(self, batch: mx.io.DataBatch, metric: mx.metric.EvalMetric):
        """
        Runs forward/backward pass and updates training metric(s).
        """
        self.module.forward_backward(batch)
        self.module.update_metric(metric, batch.label)

    def update(self):
        """
        Updates parameters of the module.
        """
        self.module.update()

    def get_gradients(self) -> Dict[str, List[mx.nd.NDArray]]:
        """
        Returns a mapping of parameters names to gradient arrays. Parameter names are prefixed with the device.
        """
        # We may have None if not all parameters are optimized
        return {"dev_%d_%s" % (i, name): exe.grad_arrays[j] for i, exe in enumerate(self.executors) for j, name in
                enumerate(self.executor_group.arg_names)
                if name in self.executor_group.param_names and self.executors[0].grad_arrays[j] is not None}

    def get_global_gradient_norm(self) -> float:
        """
        Returns global gradient norm.
        """
        # average norm across executors:
        exec_norms = [global_norm([arr for arr in exe.grad_arrays if arr is not None]) for exe in self.executors]
        norm_val = sum(exec_norms) / float(len(exec_norms))
        norm_val *= self.optimizer.rescale_grad
        return norm_val

    def rescale_gradients(self, scale: float):
        """
        Rescales gradient arrays of executors by scale.
        """
        for exe in self.executors:
            for arr in exe.grad_arrays:
                if arr is None:
                    continue
                arr *= scale

    def prepare_batch(self, batch: mx.io.DataBatch):
        """
        Pre-fetches the next mini-batch.

        :param batch: The mini-batch to prepare.
        """
        self.module.prepare(batch)

    def evaluate(self, eval_iter: data_io.BaseParallelSampleIter, eval_metric: mx.metric.EvalMetric):
        """
        Resets and recomputes evaluation metric on given data iterator.
        """
        for eval_batch in eval_iter:
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(eval_metric, eval_batch.label)

    @property
    def current_module(self) -> mx.module.Module:
        # As the BucketingModule does not expose all methods of the underlying Module we need to directly access
        # the currently active module, when we use bucketing.
        return self.module._curr_module if self._bucketing else self.module

    @property
    def executor_group(self):
        return self.current_module._exec_group

    @property
    def executors(self):
        return self.executor_group.execs

    @property
    def loss(self):
        return self.model_loss

    @property
    def optimizer(self) -> Union[mx.optimizer.Optimizer, SockeyeOptimizer]:
        """
        Returns the optimizer of the underlying module.
        """
        # TODO: Push update to MXNet to expose the optimizer (Module should have a get_optimizer method)
        return self.current_module._optimizer

    def initialize_optimizer(self, config: OptimizerConfig):
        """
        Initializes the optimizer of the underlying module with an optimizer config.
        """
        self.module.init_optimizer(kvstore=config.kvstore,
                                   optimizer=config.name,
                                   optimizer_params=config.params,
                                   force_init=True)  # force init for training resumption use case

    def save_optimizer_states(self, fname: str):
        """
        Saves optimizer states to a file.

        :param fname: File name to save optimizer states to.
        """
        self.current_module.save_optimizer_states(fname)

    def load_optimizer_states(self, fname: str):
        """
        Loads optimizer states from file.

        :param fname: File name to load optimizer states from.
        """
        self.current_module.load_optimizer_states(fname)

    def initialize_parameters(self, initializer: mx.init.Initializer, allow_missing_params: bool):
        """
        Initializes the parameters of the underlying module.

        :param initializer: Parameter initializer.
        :param allow_missing_params: Whether to allow missing parameters.
        """
        self.module.init_params(initializer=initializer,
                                arg_params=self.params,
                                aux_params=self.aux_params,
                                allow_missing=allow_missing_params,
                                force_init=False)

    def log_parameters(self):
        """
        Logs information about model parameters.
        """
        arg_params, aux_params = self.module.get_params()
        total_parameters = 0
        info = []  # type: List[str]
        for name, array in sorted(arg_params.items()):
            info.append("%s: %s" % (name, array.shape))
            total_parameters += reduce(lambda x, y: x * y, array.shape)
        logger.info("Model parameters: %s", ", ".join(info))
        if self.fixed_param_names:
            logger.info("Fixed model parameters: %s", ", ".join(self.fixed_param_names))
        logger.info("Total # of parameters: %d", total_parameters)

    def save_params_to_file(self, fname: str):
        """
        Synchronizes parameters across devices, saves the parameters to disk, and updates self.params
        and self.aux_params.

        :param fname: Filename to write parameters to.
        """
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        self.aux_params = aux_params
        super().save_params_to_file(fname)

    def load_params_from_file(self, fname: str, allow_missing_params: bool = False):
        """
        Loads parameters from a file and sets the parameters of the underlying module and this model instance.

        :param fname: File name to load parameters from.
        :param allow_missing_params: If set, the given parameters are allowed to be a subset of the Module parameters.
        """
        super().load_params_from_file(fname)  # sets self.params & self.aux_params
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=allow_missing_params)

    def install_monitor(self, monitor_pattern: str, monitor_stat_func_name: str):
        """
        Installs an MXNet monitor onto the underlying module.

        :param monitor_pattern: Pattern string.
        :param monitor_stat_func_name: Name of monitor statistics function.
        """
        self._monitor = mx.monitor.Monitor(interval=C.MEASURE_SPEED_EVERY,
                                           stat_func=C.MONITOR_STAT_FUNCS.get(monitor_stat_func_name),
                                           pattern=monitor_pattern,
                                           sort=True)
        self.module.install_monitor(self._monitor)
        logger.info("Installed MXNet monitor; pattern='%s'; statistics_func='%s'",
                    monitor_pattern, monitor_stat_func_name)

    @property
    def monitor(self) -> Optional[mx.monitor.Monitor]:
        return self._monitor


def global_norm(ndarrays: List[mx.nd.NDArray]) -> float:
    # accumulate in a list, as asscalar is blocking and this way we can run the norm calculation in parallel.
    norms = [mx.nd.square(mx.nd.norm(arr)) for arr in ndarrays if arr is not None]
    return sqrt(sum(norm.asscalar() for norm in norms))


class TrainState:
    """
    Stores the state an EarlyStoppingTrainer instance.
    """

    __slots__ = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint',
                 'updates', 'samples', 'gradient_norm', 'gradients', 'metrics', 'start_tic',
                 'early_stopping_metric', 'best_metric', 'best_checkpoint']

    def __init__(self, early_stopping_metric: str) -> None:
        self.num_not_improved = 0
        self.epoch = 0
        self.checkpoint = 0
        self.best_checkpoint = 0
        self.updates = 0
        self.samples = 0
        self.gradient_norm = None  # type: Optional[float]
        self.gradients = {}  # type: Dict[str, List[mx.nd.NDArray]]
        # stores dicts of metric names & values for each checkpoint
        self.metrics = []  # type: List[Dict]
        self.start_tic = time.time()
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = C.METRIC_WORST[early_stopping_metric]
        self.best_checkpoint = 0

    def save(self, fname: str):
        """
        Saves this training state to fname.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname: str) -> 'TrainState':
        """
        Loads a training state from fname.
        """
        with open(fname, "rb") as fp:
            return pickle.load(fp)


class EarlyStoppingTrainer:
    """
    Trainer class that fits a TrainingModel using early stopping on held-out validation data.

    :param model: TrainingModel instance.
    :param optimizer_config: The optimizer configuration.
    :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
    :param source_vocabs: Source vocabulary (and optional source factor vocabularies).
    :param target_vocab: Target vocabulary.
    """

    def __init__(self,
                 model: TrainingModel,
                 optimizer_config: OptimizerConfig,
                 max_params_files_to_keep: int,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab) -> None:
        self.model = model
        self.optimizer_config = optimizer_config
        self.max_params_files_to_keep = max_params_files_to_keep
        self.tflogger = TensorboardLogger(logdir=os.path.join(model.output_dir, C.TENSORBOARD_NAME),
                                          source_vocab=source_vocabs[0],
                                          target_vocab=target_vocab)
        self.state = None  # type: Optional[TrainState]

    def fit(self,
            train_iter: data_io.BaseParallelSampleIter,
            validation_iter: data_io.BaseParallelSampleIter,
            early_stopping_metric,
            metrics: List[str],
            checkpoint_frequency: int,
            max_num_not_improved: int,
            min_samples: Optional[int] = None,
            max_samples: Optional[int] = None,
            min_updates: Optional[int] = None,
            max_updates: Optional[int] = None,
            min_epochs: Optional[int] = None,
            max_epochs: Optional[int] = None,
            lr_decay_param_reset: bool = False,
            lr_decay_opt_states_reset: str = C.LR_DECAY_OPT_STATES_RESET_OFF,
            decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None,
            mxmonitor_pattern: Optional[str] = None,
            mxmonitor_stat_func: Optional[str] = None,
            allow_missing_parameters: bool = False,
            existing_parameters: Optional[str] = None):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder.

        :param train_iter: The training data iterator.
        :param validation_iter: The data iterator for held-out data.

        :param early_stopping_metric: The metric that is evaluated on held-out data and optimized.
        :param metrics: List of metrics that will be tracked during training.
        :param checkpoint_frequency: Frequency of checkpoints in number of update steps.

        :param max_num_not_improved: Stop training if early_stopping_metric did not improve for this many checkpoints.
               Use -1 to disable stopping based on early_stopping_metric.
        :param min_samples: Optional minimum number of samples.
        :param max_samples: Optional maximum number of samples.
        :param min_updates: Optional minimum number of update steps.
        :param max_updates: Optional maximum number of update steps.
        :param min_epochs: Optional minimum number of epochs to train, overrides early stopping.
        :param max_epochs: Optional maximum number of epochs to train, overrides early stopping.

        :param lr_decay_param_reset: Reset parameters to previous best after a learning rate decay.
        :param lr_decay_opt_states_reset: How to reset optimizer states after a learning rate decay.

        :param decoder: Optional CheckpointDecoder instance to decode and compute evaluation metrics.
        :param mxmonitor_pattern: Optional pattern to match to monitor weights/gradients/outputs
               with MXNet's monitor. Default is None which means no monitoring.
        :param mxmonitor_stat_func: Choice of statistics function to run on monitored weights/gradients/outputs
               when using MXNEt's monitor.

        :param allow_missing_parameters: Allow missing parameters when initializing model parameters from file.
        :param existing_parameters: Optional filename of existing/pre-trained parameters to initialize from.

        :return: Best score on validation data observed during training.
        """
        self._check_args(metrics, early_stopping_metric, lr_decay_opt_states_reset, lr_decay_param_reset, decoder)
        logger.info("Early stopping by optimizing '%s'", early_stopping_metric)

        self._initialize_parameters(existing_parameters, allow_missing_parameters)
        self._initialize_optimizer()

        resume_training = os.path.exists(self.training_state_dirname)
        if resume_training:
            logger.info("Found partial training in '%s'. Resuming from saved state.", self.training_state_dirname)
            utils.check_condition('dist' not in self.optimizer_config.kvstore,
                                  "Training continuation not supported with distributed training.")
            self._load_training_state(train_iter)
        else:
            self.state = TrainState(early_stopping_metric)
            self._save_params()
            self._update_best_params_link()
            self._save_training_state(train_iter)
            self._update_best_optimizer_states(lr_decay_opt_states_reset)
            self.tflogger.log_graph(self.model.current_module.symbol)
            logger.info("Training started.")

        metric_train, metric_val, metric_loss = self._create_metrics(metrics, self.model.optimizer, self.model.loss)

        process_manager = None
        if decoder is not None:
            process_manager = DecoderProcessManager(self.model.output_dir, decoder=decoder)

        if mxmonitor_pattern is not None:
            self.model.install_monitor(mxmonitor_pattern, mxmonitor_stat_func)

        speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        tic = time.time()

        next_data_batch = train_iter.next()
        while True:

            if max_epochs is not None and self.state.epoch == max_epochs:
                logger.info("Maximum # of epochs (%s) reached.", max_epochs)
                break

            if max_updates is not None and self.state.updates == max_updates:
                logger.info("Maximum # of updates (%s) reached.", max_updates)
                break

            if max_samples is not None and self.state.samples >= max_samples:
                logger.info("Maximum # of samples (%s) reached", max_samples)
                break

            ######
            # STEP
            ######
            batch = next_data_batch
            self._step(self.model, batch, checkpoint_frequency, metric_train, metric_loss)
            batch_num_samples = batch.data[0].shape[0]
            batch_num_tokens = batch.data[0].shape[1] * batch_num_samples
            self.state.updates += 1
            self.state.samples += batch_num_samples

            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()

            next_data_batch = train_iter.next()
            self.model.prepare_batch(next_data_batch)

            speedometer(self.state.epoch, self.state.updates, batch_num_samples, batch_num_tokens, metric_train)

            ############
            # CHECKPOINT
            ############
            if self.state.updates > 0 and self.state.updates % checkpoint_frequency == 0:
                time_cost = time.time() - tic
                self.state.checkpoint += 1
                # (1) save parameters and evaluate on validation data
                self._save_params()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f",
                            self.state.checkpoint, self.state.updates, self.state.epoch,
                            self.state.samples, time_cost, checkpoint_frequency / time_cost)
                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', self.state.checkpoint, name, val)
                self._evaluate(validation_iter, metric_val)
                for name, val in metric_val.get_name_value():
                    logger.info('Checkpoint [%d]\tValidation-%s=%f', self.state.checkpoint, name, val)

                # (3) update training metrics
                self._update_metrics(metric_train, metric_val, process_manager)
                metric_train.reset()

                # (4) determine improvement
                has_improved = False
                previous_best = self.state.best_metric
                for checkpoint, metric_dict in enumerate(self.state.metrics, 1):
                    value = metric_dict.get("%s-val" % early_stopping_metric, self.state.best_metric)
                    if utils.metric_value_is_better(value, self.state.best_metric, early_stopping_metric):
                        self.state.best_metric = value
                        self.state.best_checkpoint = checkpoint
                        has_improved = True

                if has_improved:
                    self._update_best_params_link()
                    self._update_best_optimizer_states(lr_decay_opt_states_reset)
                    self.state.num_not_improved = 0
                    logger.info("Validation-%s improved to %f (delta=%f).", early_stopping_metric,
                                self.state.best_metric, abs(self.state.best_metric - previous_best))
                else:
                    self.state.num_not_improved += 1
                    logger.info("Validation-%s has not improved for %d checkpoints, best so far: %f",
                                early_stopping_metric, self.state.num_not_improved, self.state.best_metric)

                # If using an extended optimizer, provide extra state information about the current checkpoint
                # Loss: optimized metric
                if metric_loss is not None and isinstance(self.model.optimizer, SockeyeOptimizer):
                    m_val = 0
                    for name, val in metric_val.get_name_value():
                        if name == early_stopping_metric:
                            m_val = val
                    checkpoint_state = CheckpointState(checkpoint=self.state.checkpoint, metric_val=m_val)
                    self.model.optimizer.pre_update_checkpoint(checkpoint_state)

                # (5) adjust learning rates
                self._adjust_learning_rate(has_improved, lr_decay_param_reset, lr_decay_opt_states_reset)

                # (6) save training state
                self._save_training_state(train_iter)

                # (7) determine stopping
                if 0 <= max_num_not_improved <= self.state.num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, self.state.num_not_improved)
                    stop_fit = True

                    if min_epochs is not None and self.state.epoch < min_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_epochs, self.state.epoch)
                        stop_fit = False

                    if min_updates is not None and self.state.updates < min_updates:
                        logger.info("Minimum number of updates (%d) not reached yet: %d",
                                    min_updates, self.state.updates)
                        stop_fit = False

                    if min_samples is not None and self.state.samples < min_samples:
                        logger.info("Minimum number of samples (%d) not reached yet: %d",
                                    min_samples, self.state.samples)

                    if stop_fit:
                        break

                tic = time.time()

        self._cleanup(lr_decay_opt_states_reset, process_manager=process_manager)
        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.state.best_checkpoint, early_stopping_metric, self.state.best_metric)
        return self.state.best_metric

    def _step(self,
              model: TrainingModel,
              batch: mx.io.DataBatch,
              checkpoint_frequency: int,
              metric_train: mx.metric.EvalMetric,
              metric_loss: Optional[mx.metric.EvalMetric] = None):
        """
        Performs an update to model given a batch and updates metrics.
        """

        if model.monitor is not None:
            model.monitor.tic()

        ####################
        # Forward & Backward
        ####################
        model.run_forward_backward(batch, metric_train)

        ####################
        # Gradient rescaling
        ####################
        gradient_norm = None
        if self.state.updates > 0 and (self.state.updates + 1) % checkpoint_frequency == 0:
            # compute values for logging to metrics (before rescaling...)
            gradient_norm = self.state.gradient_norm = model.get_global_gradient_norm()
            self.state.gradients = model.get_gradients()

        # note: C.GRADIENT_CLIPPING_TYPE_ABS is handled by the mxnet optimizer directly
        if self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_NORM:
            if gradient_norm is None:
                gradient_norm = model.get_global_gradient_norm()
            # clip gradients
            if gradient_norm > self.optimizer_config.gradient_clipping_threshold:
                ratio = self.optimizer_config.gradient_clipping_threshold / gradient_norm
                model.rescale_gradients(ratio)

        # If using an extended optimizer, provide extra state information about the current batch
        optimizer = model.optimizer
        if metric_loss is not None and isinstance(optimizer, SockeyeOptimizer):
            # Loss for this batch
            metric_loss.reset()
            metric_loss.update(batch.label, model.module.get_outputs())
            [(_, m_val)] = metric_loss.get_name_value()
            batch_state = BatchState(metric_val=m_val)
            optimizer.pre_update_batch(batch_state)

        ########
        # UPDATE
        ########
        model.update()

        if model.monitor is not None:
            results = model.monitor.toc()
            if results:
                for _, k, v in results:
                    logger.info('Monitor: Batch [{:d}] {:s} {:s}'.format(self.state.updates, k, v))

    def _evaluate(self, val_iter: data_io.BaseParallelSampleIter, val_metric: mx.metric.EvalMetric):
        """
        Evaluates the model on the validation data and updates the validation metric(s).
        """
        val_iter.reset()
        val_metric.reset()
        self.model.evaluate(val_iter, val_metric)

    def _update_metrics(self,
                        metric_train: mx.metric.EvalMetric,
                        metric_val: mx.metric.EvalMetric,
                        process_manager: Optional['DecoderProcessManager'] = None):
        """
        Updates metrics for current checkpoint. If a process manager is given, also collects previous decoding results
        and spawns a new decoding process.
        Writes all metrics to the metrics file and optionally logs to tensorboard.
        """
        checkpoint_metrics = {"epoch": self.state.epoch,
                              "learning-rate": self.model.optimizer.learning_rate,
                              "gradient-norm": self.state.gradient_norm,
                              "time-elapsed": time.time() - self.state.start_tic}
        gpu_memory_usage = utils.get_gpu_memory_usage(self.model.context)
        checkpoint_metrics['used-gpu-memory'] = sum(v[0] for v in gpu_memory_usage.values())

        for name, value in metric_train.get_name_value():
            checkpoint_metrics["%s-train" % name] = value
        for name, value in metric_val.get_name_value():
            checkpoint_metrics["%s-val" % name] = value

        if process_manager is not None:
            result = process_manager.collect_results()
            if result is not None:
                decoded_checkpoint, decoder_metrics = result
                self.state.metrics[decoded_checkpoint - 1].update(decoder_metrics)
                self.tflogger.log_metrics(decoder_metrics, decoded_checkpoint)
            process_manager.start_decoder(self.state.checkpoint)

        self.state.metrics.append(checkpoint_metrics)
        utils.write_metrics_file(self.state.metrics, self.metrics_fname)

        tf_metrics = checkpoint_metrics.copy()
        tf_metrics.update({"%s_grad" % n: v for n, v in self.state.gradients.items()})
        tf_metrics.update(self.model.params)
        self.tflogger.log_metrics(metrics=tf_metrics, checkpoint=self.state.checkpoint)

    def _cleanup(self, lr_decay_opt_states_reset: str, process_manager: Optional['DecoderProcessManager'] = None):
        """
        Cleans parameter files, training state directory and waits for remaining decoding processes.
        """
        utils.cleanup_params_files(self.model.output_dir, self.max_params_files_to_keep,
                                   self.state.checkpoint, self.state.best_checkpoint)
        if process_manager is not None:
            result = process_manager.collect_results()
            if result is not None:
                decoded_checkpoint, decoder_metrics = result
                self.state.metrics[decoded_checkpoint - 1].update(decoder_metrics)
                self.tflogger.log_metrics(decoder_metrics, decoded_checkpoint)
            utils.write_metrics_file(self.state.metrics, self.metrics_fname)

        final_training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(final_training_state_dirname):
            shutil.rmtree(final_training_state_dirname)
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            best_opt_states_fname = os.path.join(self.model.output_dir, C.OPT_STATES_BEST)
            if os.path.exists(best_opt_states_fname):
                os.remove(best_opt_states_fname)

    def _initialize_parameters(self, params: Optional[str], allow_missing_params: bool):
        self.model.initialize_parameters(self.optimizer_config.initializer, allow_missing_params)
        if params is not None:
            logger.info("Training will start with parameters loaded from '%s'", params)
            self.model.load_params_from_file(params, allow_missing_params=allow_missing_params)
        self.model.log_parameters()

    def _initialize_optimizer(self):
        self.model.initialize_optimizer(self.optimizer_config)

    def _adjust_learning_rate(self, has_improved: bool, lr_decay_param_reset: bool, lr_decay_opt_states_reset: str):
        """
        Adjusts the optimizer learning rate if required.
        """
        if self.optimizer_config.lr_scheduler is not None:
            if issubclass(type(self.optimizer_config.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler):
                lr_adjusted = self.optimizer_config.lr_scheduler.new_evaluation_result(has_improved)  # type: ignore
            else:
                lr_adjusted = False
            if lr_adjusted and not has_improved:
                if lr_decay_param_reset:
                    logger.info("Loading parameters from last best checkpoint: %d",
                                self.state.best_checkpoint)
                    self.model.load_params_from_file(self.best_params_fname)
                if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
                    logger.info("Loading initial optimizer states")
                    self.model.load_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_INITIAL))
                elif lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
                    logger.info("Loading optimizer states from best checkpoint: %d",
                                self.state.best_checkpoint)
                    self.model.load_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_BEST))

    @property
    def best_params_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.PARAMS_BEST_NAME)

    @property
    def current_params_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.PARAMS_NAME % self.state.checkpoint)

    @property
    def metrics_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.METRICS_NAME)

    @property
    def training_state_dirname(self) -> str:
        return os.path.join(self.model.output_dir, C.TRAINING_STATE_DIRNAME)

    @staticmethod
    def _create_eval_metric(metric_name: str) -> mx.metric.EvalMetric:
        """
        Creates an EvalMetric given a metric names.
        """
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        if metric_name == C.ACCURACY:
            return utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])
        elif metric_name == C.PERPLEXITY:
            return mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])
        else:
            raise ValueError("unknown metric name")

    @staticmethod
    def _create_eval_metric_composite(metric_names: List[str]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = [EarlyStoppingTrainer._create_eval_metric(metric_name) for metric_name in metric_names]
        return mx.metric.create(metrics)

    def _create_metrics(self, metrics: List[str], optimizer: mx.optimizer.Optimizer,
                        loss: loss.Loss) -> Tuple[mx.metric.EvalMetric,
                                                  mx.metric.EvalMetric,
                                                  Optional[mx.metric.EvalMetric]]:
        metric_train = self._create_eval_metric_composite(metrics)
        metric_val = self._create_eval_metric_composite(metrics)
        # If optimizer requires it, track loss as metric
        if isinstance(optimizer, SockeyeOptimizer):
            if optimizer.request_optimized_metric:
                metric_loss = self._create_eval_metric(self.state.early_stopping_metric)
            else:
                metric_loss = loss.create_metric()
        else:
            metric_loss = None
        return metric_train, metric_val, metric_loss

    def _update_best_params_link(self):
        """
        Updates the params.best link to the latest best parameter file.
        """
        best_params_path = self.best_params_fname
        actual_best_params_fname = C.PARAMS_NAME % self.state.best_checkpoint
        if os.path.lexists(best_params_path):
            os.remove(best_params_path)
        os.symlink(actual_best_params_fname, best_params_path)

    def _update_best_optimizer_states(self, lr_decay_opt_states_reset: str):
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            self.model.save_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_BEST))

    def _save_initial_optimizer_states(self, lr_decay_opt_states_reset: str):
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
            self.model.save_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_INITIAL))

    def _check_args(self,
                    metrics: List[str],
                    early_stopping_metric: str,
                    lr_decay_opt_states_reset: str,
                    lr_decay_param_reset: bool,
                    cp_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None):
        """
        Helper function that checks various configuration compatibilities.
        """
        utils.check_condition(len(metrics) > 0, "At least one metric must be provided.")
        for metric in metrics:
            utils.check_condition(metric in C.METRICS, "Unknown metric to track during training: %s" % metric)

        if 'dist' in self.optimizer_config.kvstore:
            # In distributed training the optimizer will run remotely. For eve we however need to pass information about
            # the loss, which is not possible anymore by means of accessing self.module._curr_module._optimizer.
            utils.check_condition(self.optimizer_config.name != C.OPTIMIZER_EVE,
                                  "Eve optimizer not supported with distributed training.")
            utils.check_condition(
                not issubclass(type(self.optimizer_config.lr_scheduler),
                               lr_scheduler.AdaptiveLearningRateScheduler),
                "Adaptive learning rate schedulers not supported with a dist kvstore. "
                "Try a fixed schedule such as %s." % C.LR_SCHEDULER_FIXED_RATE_INV_SQRT_T)
            utils.check_condition(not lr_decay_param_reset, "Parameter reset when the learning rate decays not "
                                                            "supported with distributed training.")
            utils.check_condition(lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_OFF,
                                  "Optimizer state reset when the learning rate decays "
                                  "not supported with distributed training.")

        utils.check_condition(self.optimizer_config.gradient_clipping_type in C.GRADIENT_CLIPPING_TYPES,
                              "Unknown gradient clipping type %s" % self.optimizer_config.gradient_clipping_type)

        utils.check_condition(early_stopping_metric in C.METRICS,
                              "Unsupported early-stopping metric: %s" % early_stopping_metric)
        if early_stopping_metric in C.METRICS_REQUIRING_DECODER:
            utils.check_condition(cp_decoder is not None, "%s requires CheckpointDecoder" % early_stopping_metric)

    def _save_params(self):
        """
        Saves model parameters at current checkpoint and optionally cleans up older parameter files to save disk space.
        """
        self.model.save_params_to_file(self.current_params_fname)
        utils.cleanup_params_files(self.model.output_dir, self.max_params_files_to_keep, self.state.checkpoint,
                                   self.state.best_checkpoint)

    def _save_training_state(self, train_iter: data_io.BaseParallelSampleIter):
        """
        Saves current training state.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)

        # (1) Parameters: link current file
        params_base_fname = C.PARAMS_NAME % self.state.checkpoint
        params_file = os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME)
        if os.path.exists(params_file):
            os.unlink(params_file)
        os.symlink(os.path.join("..", params_base_fname), params_file)

        # (2) Optimizer states
        opt_state_fname = os.path.join(training_state_dirname, C.OPT_STATES_LAST)
        self.model.save_optimizer_states(opt_state_fname)

        # (3) Data iterator
        train_iter.save_state(os.path.join(training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # (4) Random generators
        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp)

        # (5) Training state
        self.state.save(os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # (6) Learning rate scheduler
        with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
            pickle.dump(self.optimizer_config.lr_scheduler, fp)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(self.training_state_dirname):
            os.rename(self.training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, self.training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            shutil.rmtree(delete_training_state_dirname)

    def _load_training_state(self, train_iter: data_io.BaseParallelSampleIter):
        """
        Loads the full training state from disk.

        :param train_iter: training data iterator.
        """
        # (1) Parameters
        params_fname = os.path.join(self.training_state_dirname, C.TRAINING_STATE_PARAMS_NAME)
        self.model.load_params_from_file(params_fname)

        # (2) Optimizer states
        opt_state_fname = os.path.join(self.training_state_dirname, C.OPT_STATES_LAST)
        self.model.load_optimizer_states(opt_state_fname)

        # (3) Data Iterator
        train_iter.load_state(os.path.join(self.training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # (4) Random generators
        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(self.training_state_dirname, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            np.random.set_state(pickle.load(fp))

        # (5) Training state
        self.state = TrainState.load(os.path.join(self.training_state_dirname, C.TRAINING_STATE_NAME))

        # (6) Learning rate scheduler
        with open(os.path.join(self.training_state_dirname, C.SCHEDULER_STATE_NAME), "rb") as fp:
            self.optimizer_config.set_lr_scheduler(pickle.load(fp))
        # initialize optimizer again
        self._initialize_optimizer()


class TensorboardLogger:
    """
    Thin wrapper for MXBoard API to log training events.
    Flushes logging events to disk every 60 seconds.

    :param logdir: Directory to write Tensorboard event files to.
    :param source_vocab: Optional source vocabulary to log source embeddings.
    :param target_vocab: Optional target vocabulary to log target and output embeddings.
    """

    def __init__(self,
                 logdir: str,
                 source_vocab: Optional[vocab.Vocab] = None,
                 target_vocab: Optional[vocab.Vocab] = None) -> None:
        self.logdir = logdir
        self.source_labels = vocab.get_ordered_tokens_from_vocab(source_vocab) if source_vocab is not None else None
        self.target_labels = vocab.get_ordered_tokens_from_vocab(target_vocab) if source_vocab is not None else None
        try:
            import mxboard
            logger.info("Logging training events for Tensorboard at '%s'", self.logdir)
            self.sw = mxboard.SummaryWriter(logdir=self.logdir, flush_secs=60, verbose=False)
        except ImportError:
            logger.info("mxboard not found. Consider 'pip install mxboard' to log events to Tensorboard.")
            self.sw = None

    def log_metrics(self, metrics: Dict[str, Union[float, int, mx.nd.NDArray]], checkpoint: int):
        if self.sw is None:
            return

        for name, value in metrics.items():
            if isinstance(value, mx.nd.NDArray):
                self.sw.add_histogram(tag=name, values=value, bins=100, global_step=checkpoint)
            else:
                self.sw.add_scalar(tag=name, value=value, global_step=checkpoint)

    def log_graph(self, symbol: mx.sym.Symbol):
        if self.sw is None:
            return
        self.sw.add_graph(symbol)

    def log_source_embedding(self, embedding: mx.nd.NDArray, checkpoint: int):
        if self.sw is None or self.source_labels is None:
            return
        self.sw.add_embedding(tag="source", embedding=embedding, labels=self.source_labels, global_step=checkpoint)

    def log_target_embedding(self, embedding: mx.nd.NDArray, checkpoint: int):
        if self.sw is None or self.target_labels is None:
            return
        self.sw.add_embedding(tag="target", embedding=embedding, labels=self.target_labels, global_step=checkpoint)

    def log_output_embedding(self, embedding: mx.nd.NDArray, checkpoint: int):
        if self.sw is None or self.target_labels is None:
            return
        self.sw.add_embedding(tag="output", embedding=embedding, labels=self.target_labels, global_step=checkpoint)


class Speedometer:
    """
    Custom Speedometer to log samples and words per second.
    """

    def __init__(self, frequency: int = 50, auto_reset: bool = True) -> None:
        self.frequency = frequency
        self.init = False
        self.tic = 0.0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.samples = 0
        self.tokens = 0
        self.msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec %.2f tokens/sec %.2f updates/sec'

    def __call__(self, epoch: int, updates: int, samples: int, tokens: int, metric: Optional[mx.metric.EvalMetric]):
        count = updates
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.samples += samples
        self.tokens += tokens

        if self.init:
            if count % self.frequency == 0:
                toc = (time.time() - self.tic)
                updates_per_sec = self.frequency / toc
                samples_per_sec = self.samples / toc
                tokens_per_sec = self.tokens / toc
                self.samples = 0
                self.tokens = 0

                if metric is not None:
                    name_value = metric.get_name_value()
                    if self.auto_reset:
                        metric.reset()
                    logger.info(self.msg + '\t%s=%f' * len(name_value),
                                epoch, count, samples_per_sec, tokens_per_sec, updates_per_sec, *sum(name_value, ()))
                else:
                    logger.info(self.msg, epoch, count, samples_per_sec)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class DecoderProcessManager(object):
    """
    Thin wrapper around a CheckpointDecoder instance to start non-blocking decodes and collect the results.

    :param output_folder: Folder where decoder outputs are written to.
    :param decoder: CheckpointDecoder instance.
    """

    def __init__(self,
                 output_folder: str,
                 decoder: checkpoint_decoder.CheckpointDecoder) -> None:
        self.output_folder = output_folder
        self.decoder = decoder
        self.ctx = mp.get_context('spawn')  # type: ignore
        self.decoder_metric_queue = self.ctx.Queue()
        self.decoder_process = None  # type: Optional[mp.Process]

    def start_decoder(self, checkpoint: int):
        """
        Starts a new CheckpointDecoder process and returns. No other process may exist.

        :param checkpoint: The checkpoint to decode.
        """
        assert self.decoder_process is None
        output_name = os.path.join(self.output_folder, C.DECODE_OUT_NAME % checkpoint)
        self.decoder_process = self.ctx.Process(target=_decode_and_evaluate,
                                                args=(self.decoder, checkpoint, output_name, self.decoder_metric_queue))
        self.decoder_process.name = 'Decoder-%d' % checkpoint
        logger.info("Starting process: %s", self.decoder_process.name)
        self.decoder_process.start()

    def collect_results(self) -> Optional[Tuple[int, Dict[str, float]]]:
        """
        Returns the decoded checkpoint and the decoder metrics or None if the queue is empty.
        """
        self.wait_to_finish()
        if self.decoder_metric_queue.empty():
            return None
        decoded_checkpoint, decoder_metrics = self.decoder_metric_queue.get()
        assert self.decoder_metric_queue.empty()
        logger.info("Decoder-%d finished: %s", decoded_checkpoint, decoder_metrics)
        return decoded_checkpoint, decoder_metrics

    def wait_to_finish(self):
        if self.decoder_process is None:
            return
        if not self.decoder_process.is_alive():
            self.decoder_process = None
            return
        name = self.decoder_process.name
        logger.warning("Waiting for process %s to finish.", name)
        wait_start = time.time()
        self.decoder_process.join()
        self.decoder_process = None
        wait_time = int(time.time() - wait_start)
        logger.warning("Had to wait %d seconds for the Checkpoint %s to finish. Consider increasing the "
                       "checkpoint frequency (updates between checkpoints, see %s) or reducing the size of the "
                       "validation samples that are decoded (see %s)." % (wait_time, name,
                                                                          C.TRAIN_ARGS_CHECKPOINT_FREQUENCY,
                                                                          C.TRAIN_ARGS_MONITOR_BLEU))


def _decode_and_evaluate(decoder: checkpoint_decoder.CheckpointDecoder,
                         checkpoint: int,
                         output_name: str,
                         queue: mp.Queue):
    """
    Decodes and evaluates using given checkpoint_decoder and puts result in the queue,
    indexed by the checkpoint.
    """
    metrics = decoder.decode_and_evaluate(checkpoint, output_name)
    queue.put((checkpoint, metrics))
