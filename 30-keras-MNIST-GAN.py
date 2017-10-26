#!/usr/bin/python3
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten
from keras.layers import Conv2DTranspose, UpSampling2D
from keras.layers import Conv2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import matplotlib.pyplot as plt
import os

image_size = [28, 28, 1]
noise_vector_shape = [4, 4, 1]

bn_momentum = 0.9  # batch normalization momentum
lr_alpha = 0.4  # leaky ReLu alpha
dropout = 0.4
discriminator_lr = 0.0004  # discriminator learning rate
generator_lr = 0.0002  # generator learning rate

num_epochs = 20
batch_size = 128
# train_size_d = 10000
output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


def main():
    # build the GAN
    # --- generator layers --- #
    generator_input = Input(shape=noise_vector_shape, name='generator_input')
    layer = Conv2D(32, (3, 3), activation='relu', padding='same')(generator_input)

    layer = UpSampling2D((2, 2))(layer)
    layer = Conv2D(16, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization(momentum=bn_momentum)(layer)

    layer = UpSampling2D((2, 2))(layer)
    layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization(momentum=bn_momentum)(layer)

    layer = UpSampling2D((2, 2))(layer)
    generator_output = Conv2D(1, (5, 5), activation='tanh', name='generator_output')(
        layer)  # at this point image is of dim 1x28x28

    # --- discriminator layers --- #
    # need to use sequential api to reuse all layers as one model
    discriminator_input = Input(image_size, name='discriminator_input')
    layer = Conv2D(2, (3, 3), padding='same')(discriminator_input)
    layer = LeakyReLU(alpha=lr_alpha)(
        layer)  # leaky ReLU must be applied as layer instead of activation parameter to save the model
    layer = BatchNormalization(momentum=bn_momentum)(layer)
    layer = Dropout(dropout)(layer)
    layer = AveragePooling2D((2, 2))(layer)

    layer = Conv2D(8, (3, 3), padding='same')(layer)
    layer = LeakyReLU(alpha=lr_alpha)(layer)
    layer = BatchNormalization(momentum=bn_momentum)(layer)
    layer = Dropout(dropout)(layer)
    layer = AveragePooling2D((2, 2))(layer)

    layer = Conv2D(16, (3, 3), padding='same')(layer)
    layer = LeakyReLU(alpha=lr_alpha)(layer)
    layer = BatchNormalization(momentum=bn_momentum)(layer)
    layer = Dropout(dropout)(layer)
    layer = AveragePooling2D((2, 2))(layer)

    discriminator_output = Dense(1, activation='sigmoid', name='discriminator_output')(
        Flatten()(layer))  # 0 = real; 1 = fake

    # --- gan(generator indirectly learns from discriminator) and discriminator models --- #
    discriminator_optimizer = Adam(lr=discriminator_lr)
    discriminator = Model(discriminator_input, discriminator_output)  # discriminator model
    discriminator.summary()
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    generator = Model(generator_input, generator_output)  # generator model

    # gan_output = discriminator(generator_output)  # gan model which passes generator output to discriminator model
    gan_optimizer = Adam(lr=generator_lr)
    # gan = Model([generator_input, discriminator_input], [gan_output, discriminator_output])
    gan = Sequential()
    for layer in generator.layers:
        gan.add(layer)
    for layer in discriminator.layers[1:]:
        gan.add(layer)
    gan.summary()
    gan.compile(gan_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (*x_train.shape, 1))
    x_test = np.reshape(x_test, (*x_test.shape, 1))

    num_batches = int(len(x_train) / batch_size)

    # def data_generator(x_data, y_data, batch_size):
    #     # preprocess image
    #     train_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=False)
    #     train_data.fit(x_data)
    #     for x_batch, y_batch in train_data.flow(x_data, y_data, batch_size=batch_size):
    #         if x_batch.shape[0] < batch_size:
    #             print("\nLast batch, so actual batch size = {} instead of {}".format(x_batch.shape[0], batch_size))
    #         # sample a noise batch
    #         noise_samples = np.random.normal(0, 1.0, size=[x_batch.shape[0]] + noise_vector_shape)
    #         batch_input = {'generator_input': noise_samples}
    #         batch_output = [np.ones(y_batch.shape)]
    #         yield (batch_input, batch_output)

    # training
    # # -- first we train the discriminator alone for a while -- #
    # history = discriminator.fit(x_train_d, y_train_d, epochs=1, batch_size=32)
    # print("DISCRIMINATOR PRELIMINARY TRAINING DONE")
    # print(history.history)
    #
    # # --- set callbacks --- #
    # checkpoint = ModelCheckpoint(filepath='mnist_gan_model_epoch_{epoch:02d}.hdf5', verbose=1, monitor='val_loss')
    # tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    # # --- perform training --- #
    # history = gan.fit_generator(data_generator(x_train, y_train, batch_size),
    #                             steps_per_epoch=num_batches,
    #                             epochs=num_epochs,
    #                             validation_data=data_generator(x_test, y_test, 32),
    #                             validation_steps=10,
    #                             callbacks=[checkpoint, tb])
    # print(history.history)

    # preprocess image
    train_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=False)
    train_data.fit(x_train)

    for i in range(num_epochs):
        print("Epoch {}/{}".format(i + 1, num_epochs))
        batch_count = 0
        for x_batch, y_batch in train_data.flow(x_train, y_train, batch_size=batch_size):
            if x_batch.shape[0] < batch_size:
                print("\nLast batch, so actual batch size = {} instead of {}".format(x_batch.shape[0], batch_size))
            # sample a noise batch for gan
            noise_samples = np.random.normal(0, 1.0, size=[x_batch.shape[0]] + noise_vector_shape)
            batch_input = noise_samples
            batch_output = np.ones(y_batch.shape)

            # sample discriminator training batch
            batch_size_half = int(x_batch.shape[0] / 2)
            train_indices = np.random.randint(0, x_train.shape[0],
                                              batch_size_half)  # randomly sample batch from from training set
            noise_samples = np.random.normal(0, 1.0, size=[batch_size_half] + noise_vector_shape)
            x_train_g = generator.predict(noise_samples)
            x_train_d = np.concatenate((x_train[train_indices, :, :, :], x_train_g), axis=0)
            y_train_d = np.zeros((x_train_d.shape[0],))
            y_train_d[batch_size_half:, ] = 1

            # train discriminator
            discriminator.train_on_batch(x_train_d, y_train_d)
            # train gan (implicitly generator)
            gan.train_on_batch(batch_input, batch_output)

            batch_count += 1
            print("Trained batch {}/{}".format(batch_count, num_batches), end='\r')
            if batch_count >= num_batches:
                break

        # test generator #
        noise_samples = np.random.normal(0, 1.0, size=[10] + noise_vector_shape)
        output_images = generator.predict(noise_samples)
        # rescale to 0-1 range
        output_images -= output_images.min()
        output_images /= output_images.max()
        # tile them for 3-channel output
        output_images = np.tile(output_images, (1, 1, 3))
        for k in range(output_images.shape[0]):
            plt.imsave(os.path.join(output_dir, 'output_epoch_{}_{}.png'.format(i+1, k + 1)), output_images[k])


if __name__ == "__main__":
    main()