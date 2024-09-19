"""Classes that define the generator and the discriminator.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)


class DCGAN_Generator(object):
    def __init__(self, batch_size, noize_dim=100):
        self.batch_size = batch_size
        self.noize_dim = noize_dim
        self.w_init = RandomNormal(mean=0.0, stddev=0.02)

    def build(self):
        noize = Input(batch_shape=(self.batch_size, self.noize_dim))

        densed = Dense(4 * 4 * 1024, "relu", kernel_initializer=self.w_init)(noize)
        densed = BatchNormalization()(densed)
        reshaped = Reshape((4, 4, 1024))(densed)

        # 引数：(チャンネル数、カーネルサイズ、ストライド、活性化関数)
        conv_1 = Conv2DTranspose(512, (5, 5), (2, 2), "same", activation="relu", kernel_initializer=self.w_init)(reshaped)
        conv_1 = BatchNormalization()(conv_1)
        conv_2 = Conv2DTranspose(256, (5, 5), (2, 2), "same", activation="relu", kernel_initializer=self.w_init)(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_3 = Conv2DTranspose(128, (5, 5), (2, 2), "same", activation="relu", kernel_initializer=self.w_init)(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        conv_4 = Conv2DTranspose(3, (5, 5), (2, 2), "same", activation="tanh", kernel_initializer=self.w_init)(conv_3)

        generator = Model(inputs=noize, outputs=conv_4)

        return generator


class DCGAN_Discriminator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.w_init = RandomNormal(mean=0.0, stddev=0.02)

    def build(self):
        images = Input(batch_shape=(self.batch_size, 64, 64, 3))
        conv_1 = Conv2D(128, (5, 5), (2, 2), "same", kernel_initializer=self.w_init)(images)
        conv_1 = LeakyReLU(alpha=0.2)(conv_1)

        conv_2 = Conv2D(256, (5, 5), (2, 2), "same", kernel_initializer=self.w_init)(conv_1)
        conv_2 = LeakyReLU(alpha=0.2)(conv_2)
        conv_2 = BatchNormalization()(conv_2)

        conv_3 = Conv2D(512, (5, 5), (2, 2), "same", kernel_initializer=self.w_init)(conv_2)
        conv_3 = LeakyReLU(alpha=0.2)(conv_3)
        conv_3 = BatchNormalization()(conv_3)

        conv_4 = Conv2D(1024, (5, 5), (2, 2), "same", kernel_initializer=self.w_init)(conv_2)
        conv_4 = LeakyReLU(alpha=0.2)(conv_4)
        conv_4 = BatchNormalization()(conv_4)

        flatten = Flatten()(conv_3)
        densed = Dense(1, "sigmoid", kernel_initializer=self.w_init)(flatten)

        discriminator = Model(inputs=images, outputs=densed)
        return discriminator
