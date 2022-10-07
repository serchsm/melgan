import tensorflow as tf
import tensorflow_addons as tfa


class MelSpectrogram(tf.keras.layers.Layer):
    def __init__(
        self,
        sample_rate=22050,
        frame_length=1024,
        frame_step=256,
        fft_length=None,
        # fft_length=1024,
        num_mel_bins=80,
        f_min=125,
        # f_max=3800,
        f_max=7600,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.num_mel_bins = num_mel_bins
        self.f_min = f_min
        self.f_max = f_max
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

    def call(self, audio, training=True):
        if training:
            stfts = tf.signal.stft(
                tf.squeeze(audio, -1),
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                pad_end=True,
            )
            # stfts = tf.abs(stfts)
            mel_spectrograms = tf.matmul(
                tf.abs(stfts), self.linear_to_mel_weight_matrix
            )
            return tf.math.log(mel_spectrograms + 1e-6)
        else:
            return audio


class MelGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super(MelGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(
        self,
        optimizer_gen,
        optimizer_dis,
        feature_matching_loss,
        generator_loss,
        discriminator_loss,
    ):
        super(MelGAN, self).compile()
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis
        self.feature_matching_loss = feature_matching_loss
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.gen_loss_metric = tf.keras.metrics.Mean(name="gen_loss")
        self.dis_loss_metric = tf.keras.metrics.Mean(name="dis_loss")

    def train_step(self, batch):
        x_batch, y_batch = batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Train the Discriminator
            gen_audio = self.generator(x_batch, training=True)
            dfake_outputs = self.discriminator(gen_audio)
            dreal_outputs = self.discriminator(x_batch)
            d_loss = self.discriminator_loss(dreal_outputs, dfake_outputs)
            # Train Generator to fool the discriminator
            feat_match_loss = self.feature_matching_loss(dfake_outputs, dreal_outputs)
            gen_loss = self.generator_loss(dfake_outputs)
            final_gen_loss = gen_loss + 10.0 * feat_match_loss

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer_dis.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )
        g_grads = gen_tape.gradient(final_gen_loss, self.generator.trainable_weights)
        self.optimizer_gen.apply_gradients(
            zip(g_grads, self.generator.trainable_weights)
        )

        self.gen_loss_metric.update_state(final_gen_loss)
        self.dis_loss_metric.update_state(d_loss)

        return {
            "gen_loss": self.gen_loss_metric.result(),
            "disc_loss": self.dis_loss_metric.result(),
        }


def residual_stack(x, filters):
    residual = x
    # block
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, 3, padding="same", dilation_rate=1),
        data_init=False,
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, 3, padding="same", dilation_rate=1),
        data_init=False,
    )(x)
    add_1 = tf.keras.layers.add([x, residual])
    x = tf.keras.layers.LeakyReLU()(add_1)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, 3, padding="same", dilation_rate=3),
        data_init=False,
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, 3, padding="same", dilation_rate=1),
        data_init=False,
    )(x)
    add_2 = tf.keras.layers.add([x, add_1])
    x = tf.keras.layers.LeakyReLU()(add_2)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, 3, padding="same", dilation_rate=9),
        data_init=False,
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, 3, padding="same", dilation_rate=1),
        data_init=False,
    )(x)
    return tf.keras.layers.add([x, add_2])


def upsampling_block(x, stride, filters):
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1DTranspose(filters, 2 * stride, stride, padding="same"),
        data_init=False,
    )(x)
    return residual_stack(x, filters)


def get_generator(in_shape):
    x_in = tf.keras.Input(shape=in_shape)
    x = MelSpectrogram()(x_in)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(512, 7, padding="same"), data_init=False
    )(x)
    # Upsample 8x
    x = upsampling_block(x, 8, 256)
    # Upsample 8x
    x = upsampling_block(x, 8, 128)
    # Upsamling 2x
    x = upsampling_block(x, 2, 64)
    # Upsamling 2x
    x = upsampling_block(x, 2, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    y = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(1, 7, padding="same", activation="tanh"), data_init=False
    )(x)
    return tf.keras.Model(x_in, y, name="generator")


def discriminator_block(x):
    # TODO: Ouput feature maps
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(16, 15, padding="same"), data_init=False
    )(x)
    fmap_1 = tf.keras.layers.LeakyReLU()(x)
    # Check effect of stride here
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(64, 41, groups=4, strides=4, padding="same"),
        data_init=False,
    )(fmap_1)
    fmap_2 = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(256, 41, groups=16, strides=4, padding="same"),
        data_init=False,
    )(fmap_2)
    fmap_3 = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(1024, 41, groups=64, strides=4, padding="same"),
        data_init=False,
    )(fmap_3)
    fmap_4 = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(1024, 41, groups=256, strides=4, padding="same"),
        data_init=False,
    )(fmap_4)
    fmap_5 = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(1024, 5, strides=1, padding="same"), data_init=False
    )(fmap_5)
    fmap_6 = tf.keras.layers.LeakyReLU()(x)
    output = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(1, 3, strides=1, padding="same"), data_init=False
    )(fmap_6)
    return [fmap_1, fmap_2, fmap_3, fmap_4, fmap_5, fmap_6, output]


def create_discriminator(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    d1_outputs = discriminator_block(d_in)
    ds2x = tf.keras.layers.AveragePooling1D(pool_size=4, strides=2, padding="same")(
        d_in
    )
    d2_outputs = discriminator_block(ds2x)
    ds4x = tf.keras.layers.AveragePooling1D(pool_size=4, strides=4, padding="same")(
        ds2x
    )
    d3_outputs = discriminator_block(ds4x)
    return tf.keras.Model(
        d_in, outputs=[d1_outputs, d2_outputs, d3_outputs], name="discriminator"
    )
