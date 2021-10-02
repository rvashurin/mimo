# coding=utf-8
# Copyright 2021 The Edward2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Multi-headed simple CNN on Fashion MNIST"""
import functools
import os
import time
from absl import app
from absl import flags
from absl import logging

import fmnist_model  # local file import
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
import uncertainty_baselines as ub
import uncertainty_metrics as um

flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_float('input_repetition_probability', 0.0,
                   'The probability that the inputs are identical for the'
                   'ensemble members.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('batch_repetitions', 1, 'Number of times an example is'
                     'repeated in a training batch. More repetitions lead to'
                     'lower variance gradients and increased training time.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 0,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['80', '160', '180'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 3e-4, 'L2 coefficient.')
flags.DEFINE_string('dataset', 'fashion_mnist', help='Dataset.')
flags.DEFINE_integer(
    'checkpoint_interval', -1,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string(
    'output_dir', '/tmp/fmnist', 'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 250, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  builder = tfds.builder(FLAGS.dataset)
  ds_info = builder.info
  # 64
  train_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores // FLAGS.batch_repetitions
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

  # 60000
  train_dataset = builder.as_dataset(split='train').shuffle(buffer_size=60000)
  # 10000
  val_dataset = train_dataset.skip(50000)
  # 50000
  train_dataset = train_dataset.take(50000)
  # 10000
  test_dataset = builder.as_dataset(split='test')

  train_dataset_size = train_dataset.cardinality().numpy()
  test_dataset_size = train_dataset.cardinality().numpy()

  steps_per_epoch = train_dataset_size // train_batch_size
  steps_per_eval = test_dataset_size // test_batch_size

  num_classes = ds_info.features['label'].num_classes

  train_dataset = train_dataset.shuffle(buffer_size=50000,
                                        reshuffle_each_iteration=False).repeat(FLAGS.train_epochs).batch(train_batch_size)
  val_dataset = val_dataset.batch(test_batch_size)
  test_dataset = test_dataset.batch(test_batch_size)

  test_datasets = {
      'clean': test_dataset
  }

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras model')
#    model = fmnist_model.small_resnet(
#        input_shape=[FLAGS.ensemble_size] +
#        list(ds_info.features['image'].shape),
#        depth=10,
#        width_multiplier=1,
#        num_classes=num_classes,
#        ensemble_size=FLAGS.ensemble_size)
    model = fmnist_model.simple_resnet(40, 1, num_classes)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * train_batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(
        lr_schedule, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.SGD(base_lr)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/diversity': rm.metrics.AveragePairwiseDiversity(),
    }

    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.keras.metrics.SparseCategoricalAccuracy())

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['image']
      labels = inputs['label']
      batch_size = tf.shape(images)[0]

#      main_shuffle = tf.random.shuffle(tf.tile(
#          tf.range(batch_size), [FLAGS.batch_repetitions]))
#      to_shuffle = tf.cast(tf.cast(tf.shape(main_shuffle)[0], tf.float32)
#                           * (1. - FLAGS.input_repetition_probability),
#                           tf.int32)
#      shuffle_indices = [
#          tf.concat([tf.random.shuffle(main_shuffle[:to_shuffle]),
#                     main_shuffle[to_shuffle:]], axis=0)
#          for _ in range(FLAGS.ensemble_size)]
#      images = tf.stack([tf.gather(images, indices, axis=0)
#                         for indices in shuffle_indices], axis=1)
#      labels = tf.stack([tf.gather(labels, indices, axis=0)
#                         for indices in shuffle_indices], axis=1)

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
#        negative_log_likelihood = tf.reduce_mean(tf.reduce_sum(
#            tf.keras.losses.sparse_categorical_crossentropy(
#                labels, logits, from_logits=True)))
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True))

        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the BN parameters and bias terms.
          if ('kernel' in var.name or 'batch_norm' in var.name or
              'bias' in var.name):
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(tf.reshape(logits, [-1, num_classes]))
      flat_labels = tf.reshape(labels, [-1])
      metrics['train/ece'].update_state(flat_labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(flat_labels, probs)

    strategy.run(step_fn, args=((inputs),))

  @tf.function
  def test_step(inputs):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['image']
      labels = inputs['label']
#      images = tf.tile(
#          tf.expand_dims(images, 1), [1, FLAGS.ensemble_size, 1, 1, 1])
      logits = model(images, training=False)
      probs = tf.nn.softmax(logits)

#      per_probs = tf.transpose(probs, perm=[1, 0, 2])
#      metrics['test/diversity'].add_batch(per_probs)
#
#      for i in range(FLAGS.ensemble_size):
#        member_probs = probs[:, i]
#        member_loss = tf.keras.losses.sparse_categorical_crossentropy(
#            labels, member_probs)
#        metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
#        metrics['test/accuracy_member_{}'.format(i)].update_state(
#            labels, member_probs)
#
#      # Negative log marginal likelihood computed in a numerically-stable way.
#      labels_tiled = tf.tile(
#          tf.expand_dims(labels, 1), [1, FLAGS.ensemble_size])
#      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
#          labels_tiled, logits, from_logits=True)
#      negative_log_likelihood = tf.reduce_mean(
#          -tf.reduce_logsumexp(log_likelihoods, axis=[1]) +
#          tf.math.log(float(FLAGS.ensemble_size)))
#      probs = tf.math.reduce_mean(probs, axis=1)  # marginalize
#
#      metrics['test/negative_log_likelihood'].update_state(
#          negative_log_likelihood)
      metrics['test/accuracy'].update_state(labels, probs)
      metrics['test/ece'].update_state(labels, probs)

    strategy.run(step_fn, args=((inputs),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

  start_time = time.time()
  for current_step, inputs in enumerate(train_dataset):
    train_step(inputs)

    epoch = current_step // steps_per_epoch
    if current_step % steps_per_epoch == 0.0:
      logging.info('Starting to run epoch: %s', epoch)
    max_steps = steps_per_epoch * (FLAGS.train_epochs)
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    if current_step % 20 == 0:
      logging.info(message)

    if current_step % steps_per_epoch == 0:
      logging.info('Testing on validation dataset')
      for step, inputs in enumerate(val_dataset):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step, epoch)
        test_start_time = time.time()
        test_step(inputs)
        ms_per_example = (time.time() - test_start_time) * 1e6 / test_batch_size
        metrics['test/ms_per_example'].update_state(ms_per_example)
      logging.info('Done with testing on validation dataset')

      logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                   metrics['train/loss'].result(),
                   metrics['train/accuracy'].result() * 100)
      logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                   metrics['test/negative_log_likelihood'].result(),
                   metrics['test/accuracy'].result() * 100)
      for i in range(FLAGS.ensemble_size):
        logging.info(
            'Member %d Test Loss: %.4f, Accuracy: %.2f%%', i,
            metrics['test/nll_member_{}'.format(i)].result(),
            metrics['test/accuracy_member_{}'.format(i)].result() * 100)

      total_results = {name: metric.result() for name, metric in metrics.items()}
      # Results from Robustness Metrics themselves return a dict, so flatten them.
      total_results = utils.flatten_dictionary(total_results)
      with summary_writer.as_default():
        for name, result in total_results.items():
          tf.summary.scalar(name, result, step=epoch + 1)

      for _, metric in metrics.items():
        metric.reset_states()

      if (FLAGS.checkpoint_interval > 0 and
          (epoch + 1) % FLAGS.checkpoint_interval == 0):
        checkpoint_name = checkpoint.save(
            os.path.join(FLAGS.output_dir, 'checkpoint'))
        logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)


if __name__ == '__main__':
  with tf.device('/device:gpu:0'):
    app.run(main)
