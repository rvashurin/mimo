"""Multi-headed simple CNN on Fashion MNIST"""
import os
import math

import numpy as np
import tensorflow as tf

from tqdm import trange

import matplotlib as mpl
import matplotlib.pyplot as plt

def create_mimo(architecture, data_dim=1, ens_size=1, activation='relu'):
  """Create a MIMO model by expanding input/ouput layer by ensemble size."""
  # The only modification needed by MIMO: expand input/output layer by ensemble size
  num_logits = 1  # Since this is a regression problem.
  inputs_size = data_dim * ens_size
  outputs_size = num_logits * ens_size

  # MIMO input: expand input layer by ensemble size.
  inputs = tf.keras.layers.Input(shape=(inputs_size,))

  # Use the classic MLP encoder.
  net = inputs
  net = tf.keras.layers.Flatten()(net)
  for units in architecture:
    net = tf.keras.layers.Dense(units, activation=activation)(net)

  # MIMO output: Expand output layer by ensemble size.
  outputs = tf.keras.layers.Dense(outputs_size, activation='linear')(net)
  mimo_mlp = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  return mimo_mlp

def create_data(n, data_dim=1, data_noise=0.02, batch_size=64, support=(-1., 1.)):
  """Create regression data from Blundell et al (2016)

  Args:
    n: (int) Sample size.
    data_dim: (int) The dimensionality of the features.
      A 1D feature will first be created, and if data_dim > 1. It will be
      projected to data_dim using a random Gaussian matrix.
    data_noise: (float) The standard deviation of the observation noise in the data-generation mechanism.
    batch_size: (int) Batch size for the tf dataset to be generated.
    support: (tuple of float) Range of X to sample the 1D feature from.

  Returns:
    X0: (ndarray) Original 1D features, shape (n, 1)
    y: (ndarray) Response, shape (n, 1)
    X: (ndarray) High-dimensional features obtained by projecting X0 to high dimension, shape (n, data_dim)
    tf_dataset: (tf.data.Dataset) A TF Dataset to be used for training loops.
  """
  # Create 1D features X0.
  lower, upper = support
  X0 = tf.random.uniform((n, 1))* (upper-lower) + lower
  noise = tf.random.normal((n, 1)) * data_noise

  # Generate response.
  y = X0 + 0.3 * tf.math.sin(2*math.pi * (X0 + noise)) + 0.3 * tf.math.sin(4*math.pi * (X0 + noise)) + noise

  # Embed X0 into high-dimensional space.
  X = X0
  if data_dim > 1:
    Proj_mat = tf.random.normal(shape=(1, data_dim))
    X = tf.matmul(X0, Proj_mat)

  # Produce high-dimensional dataset.
  tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))
  tf_dataset = tf_dataset.shuffle(1024).batch(batch_size)

  return X0, y, X, tf_dataset

def get_train_step(ens_size):

  assert ens_size >= 1

  @tf.function
  def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:

      inputs = [x]
      targets = [y]

      for _ in range(ens_size-1):
        rd_index = tf.random.shuffle(tf.range(len(y)))
        shuffled_x = tf.gather(x, rd_index)
        shuffled_y = tf.gather(y, rd_index)
        inputs.append(shuffled_x)
        targets.append(shuffled_y)

      inputs = tf.concat(inputs, 1)
      targets = tf.concat(targets, 1)

      predictions = model(inputs, training=True)
      sq_loss = tf.reduce_mean(tf.square(targets-predictions))

    gradients = tape.gradient(sq_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return sq_loss

  return train_step

def run_mimo_experiments(architecture=(32, 128), ens_sizes=(1, 2, 3, 4, 5),
                         lr=0.01, batch_size=32, n_epochs=2000,
                         data_dim=1, data_noise=0.08,
                         n_train=64, n_test=3000, num_reps=20,
                         eval_epoch=2, print_epoch=-1):
  """Train MIMO model with different ensemble sizes over multiple random seeds.

  Args:
    architecture: (tuple of int) Number of units in each hidden layer the MIMO MLP model.
    ens_sizes: (tuple of int) Ensemble sizes for MIMO model to evaluate in experiments.
    lr: (float) Learning rate.
    batch_size: (int) Batch size used for training.
    n_epochs: (int) Number of training epochs.
    data_dim: (int) Dimensionality of the input features.
    data_noise: (float) The standard deviation of the observation noise in the data-generation mechanism.
    n_train: (int) Number of training examples.
    n_test: (int) Number of test examples.
    num_reps: (int) Number of repetitions to conduct for each ensemble size.
    eval_epoch: (int) Record eval results every eval_epoch.
    print_epoch: (int) Print eval results every print_epoch. Set to -1 to disable.

  Returns:
    Xtest0: (ndarray) Original 1D features for testing data with support in
      (-1., 1.), shape (n, 1)
    ytest: (ndarray) Response for testing data, shape (n, 1)
    ytest_pred: (dict) A nested dictionary storing evaluation result for each
      repetition, each ensemble size, and for specified epochs.
    Xtest0_wide: (ndarray) Similar to Xtest0 but from a testing dataset sampled
      from a wider support (-2., 2.), shape (n, 1)
    ytest_wide: (ndarray) Similar to ytest but from a wider-support test dataset,
      shape (n, 1)
    ytest_wide_pred: (dict) Similar to ytest_pred but from a wider-support test
      dataset.
  """
  tf.keras.backend.clear_session()
  tf.random.set_seed(0)

  # Create testing data.
  Xtest0, ytest, Xtest, _ = create_data(n_test, data_dim=data_dim,
                                   data_noise=data_noise, support=(-1.,1.))
  Xtest0_wide, ytest_wide, Xtest_wide, _ = create_data(n_test, data_dim=data_dim,
                                             data_noise=data_noise, support=(-2.,2.))

  # Train MIMO models with different ensemble sizes over multiple random seeds.
  ytest_pred = {}
  ytest_wide_pred = {}

  for rep in range(num_reps):
    print('Repetition', rep)

    _, y, X, training_set = create_data(n_train, data_dim=data_dim,
                                        data_noise=data_noise, batch_size=batch_size)

    ytest_pred[rep] = {}
    ytest_wide_pred[rep] = {}

    for ens_size_id in trange(len(ens_sizes)):
      # Specified ensemble size.
      ens_size = ens_sizes[ens_size_id]

      ytest_pred[rep][ens_size] = {}
      ytest_wide_pred[rep][ens_size] = {}

      # Train a MIMO model.
      optimizer = tf.keras.optimizers.Adam(lr)
      mimo_mlp = create_mimo(architecture, data_dim=data_dim,
                             ens_size=ens_size, activation='relu')
      train_step = get_train_step(ens_size)

      for epoch in range(n_epochs):
        sq_loss = []
        for x, y in training_set:
          loss = train_step(mimo_mlp, optimizer, x, y)
          sq_loss.append(loss)

        if print_epoch > 0 and epoch % print_epoch == 0:
          print('[{:4d}] train sq. loss {:0.3f}'.format(epoch, np.mean(sq_loss)))

        if epoch % eval_epoch == 0:
          # Save testing performance.
          per_ens_member_ytest_pred = mimo_mlp(tf.tile(Xtest, (1, ens_size)))
          ytest_pred[rep][ens_size][epoch] = per_ens_member_ytest_pred

          per_ens_member_ytest_wide_pred = mimo_mlp(tf.tile(Xtest_wide, (1, ens_size)))
          ytest_wide_pred[rep][ens_size][epoch] = per_ens_member_ytest_wide_pred

  return Xtest0, ytest, ytest_pred, Xtest0_wide, ytest_wide, ytest_wide_pred

experiment_config = dict(
    data_dim=1,
    architecture=(32, 128),
    n_train=64,
    n_test=3000,
    num_reps=20,
    lr=0.01,
    n_epochs=2000,
    ens_sizes=(1,2,3,4,5))

with tf.device('/GPU:1'):
    Xtest, ytest, ytest_pred, Xtest_wide, ytest_wide, ytest_wide_pred = run_mimo_experiments(**experiment_config)

# RMSE vs number of models in ens)

ens_sizes = range(1,6)
rmse = []
std = []

for ens_size in ens_sizes:
  rmses = []
  for i in range(len(ytest_pred.keys())):
    diff = ytest - ytest_pred[i][ens_size][2000]
    rmses.append(math.sqrt(np.mean(diff ** 2)))
  if ens_size == 1:
    print(min(rmses))
    print(max(rmses))
  rmse.append(np.mean(rmses))
  std.append(np.std(rmses))

rmse = np.array(rmse)
std = np.array(std)

plt.plot(ens_sizes, rmse)
plt.fill_between(ens_sizes, rmse-std, rmse+std)
plt.ylabel('RMSE')
plt.xlabel('Ensemble size')
plt.savefig('synthetic_results.png')
