import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

mae = MeanAbsoluteError()


def feature_matching_loss(dfake_outputs, dreal_outputs):
    num_dblocks = len(dfake_outputs)
    # Loss over feature maps only
    num_featmaps = len(dfake_outputs[0]) - 1
    block_loss = []
    for k in range(num_dblocks):
        featmap_loss = [
            mae(dreal_outputs[k][i], dfake_outputs[k][i]) for i in range(num_featmaps)
        ]
        block_loss.append(tf.reduce_mean(featmap_loss))
    loss_val = tf.reduce_sum(block_loss)
    return loss_val


def gen_loss(dfake_outputs):
    return tf.reduce_sum([-1.0 * tf.reduce_mean(d_k_o[-1]) for d_k_o in dfake_outputs])
    # return -1.*tf.reduce_mean([tf.reduce_sum(d_k[-1]) for d_k in dk_outputs])


def discriminator_loss(dreal_outputs, dfake_outputs):
    loss_real = 0
    loss_fake = 0
    for k in range(len(dreal_outputs)):
        loss_real += tf.reduce_mean(tf.keras.layers.ReLU()(1 - dreal_outputs[k][-1]))
        loss_fake += tf.reduce_mean(tf.keras.layers.ReLU()(1 + dfake_outputs[k][-1]))
    return loss_real + loss_fake
