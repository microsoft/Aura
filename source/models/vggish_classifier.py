import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

from .vggish import VGGish

class VGGishClassifier(Model):
    """
    Logistic type classifier using VGGIsh as a feature extractor
    """

    def __init__(self, extract_init=None, nclass=520, output_bias=None):
        super().__init__()

        self.vggish = VGGish()
        if extract_init is not None:
            self.vggish.load_weights(extract_init)

        self.eps = 10**-8

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(128, activation='relu', name='embedding_layer')
        self.final = Dense(nclass, activation='sigmoid', bias_initializer=output_bias)

        self.dropout = Dropout(rate=0.3)
        self.attention = Dense(nclass, activation='softmax')


    def estimate(self, x):

        x = tf.expand_dims(x, axis=3)
        _, embed = self.vggish(x)

        embed = tf.reduce_mean(embed, 2)

        x = self.dense1(embed)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        out = self.final(x) #(B, time, nlabels)
        att = self.attention(x) #(B, time, units)

        att = tf.clip_by_value(att, self.eps, 1. - self.eps)
        att_norm = tf.reduce_sum(att, 1, keepdims=True)

        out = tf.reduce_sum(out * att / att_norm, 1)

        return out, x

    def call(self, x):
        out, latent = self.estimate(x)

        return out