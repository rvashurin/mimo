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
"""Wide ResNet architecture with multiple input and outputs."""
import functools
import layers  # local file import
import tensorflow as tf

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=7,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def basic_block(inputs, filters, strides):
  """Basic residual block of two 3x3 convs."""

  x = inputs
  y = inputs
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=strides)(y)
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=1)(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters, kernel_size=1, strides=strides)(x)

  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


def multiheaded_resnet(input_shape, depth, width_multiplier, num_classes,
                ensemble_size):
  """Builds Wide ResNet with Sparse BatchEnsemble.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor. The input shape must be (ensemble_size, width,
      height, channels).
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    ensemble_size: Number of ensemble members.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  input_shape = list(input_shape)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.Permute([2, 3, 4, 1])(inputs)
  if ensemble_size != input_shape[0]:
    raise ValueError('the first dimension of input_shape must be ensemble_size')
  x = tf.keras.layers.Reshape(input_shape[1:-1] +
                              [input_shape[-1] * ensemble_size])(x)
  x = Conv2D(16, strides=1)(x)
  for strides, filters in zip([1, 3, 1], [16, 32, 64]):
    x = group(
        x,
        filters=filters * width_multiplier,
        strides=strides,
        num_blocks=num_blocks)

  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=6)(x)
  x = tf.keras.layers.Flatten()(x)
  x = layers.DenseMultihead(
      num_classes,
      kernel_initializer='he_normal',
      activation=None,
      ensemble_size=ensemble_size)(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def deep_multiheaded_resnet(input_shape, depth, width_multiplier, num_classes,
                ensemble_size):
  """Builds Wide ResNet with Sparse BatchEnsemble.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor. The input shape must be (ensemble_size, width,
      height, channels).
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    ensemble_size: Number of ensemble members.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  input_shape = list(input_shape)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.Permute([2, 3, 4, 1])(inputs)
  if ensemble_size != input_shape[0]:
    raise ValueError('the first dimension of input_shape must be ensemble_size')
  x = tf.keras.layers.Reshape(input_shape[1:-1] +
                              [input_shape[-1] * ensemble_size])(x)
  x = Conv2D(16, strides=1)(x)
  for strides, filters in zip([1, 3, 1], [16, 32, 64]):
    x = group(
        x,
        filters=filters * width_multiplier,
        strides=strides,
        num_blocks=num_blocks)

  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=6)(x)
  x = tf.keras.layers.Flatten()(x)

  x = tf.keras.layers.Dense(num_classes * ensemble_size, activation='relu')(x)
  xs = {}
  for i in range(ensemble_size):
    start_idx = i * num_classes
    end_idx = start_idx + num_classes
    xs[i] = x[:, start_idx:end_idx]
  for i in xs:
      xs[i] = tf.keras.layers.Dense(num_classes, activation=None)(xs[i])
  x = tf.keras.layers.concatenate([xs[i] for i in xs], axis=1)
  batch_size = tf.shape(inputs)[0]
  outputs = tf.reshape(x, [batch_size,
                           ensemble_size,
                           num_classes])

  return tf.keras.Model(inputs=inputs, outputs=outputs)

def simple_resnet(depth, width_multiplier, num_classes):
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=(28, 28, 1))
  x = Conv2D(16, strides=2)(inputs)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(
        x,
        filters=filters * width_multiplier,
        strides=strides,
        num_blocks=num_blocks)

  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=4)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(num_classes, activation=None)(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def extremely_simple_net(num_classes):
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                               strides=(1, 1), padding='same',
                               data_format='channels_last',
                               name='conv_1', activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                               strides=(1, 1), padding='same',
                               data_format='channels_last',
                               name='conv_2', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(num_classes, activation=None)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
