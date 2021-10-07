import tensorflow as tf
import numpy as np


class Attention(tf.keras.layers.Layer):
    def __init__(self, clip=None, return_logits=False, head_depth=16, inf=1e+10, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.return_logits = return_logits
        self.inf = inf
        dk = tf.cast(head_depth, tf.float32)
        self.scale = tf.math.sqrt(dk)

    def call(self, x, mask=None):
        """ Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
            K: (batch, n_heads, k_seq(=n_nodes), head_depth)
            logits: (batch, n_heads, q_seq(this could be 1), k_seq)
            mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
            mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
            [True] -> [1 * -np.inf], [False] -> [logits]
        """
        Q, K, V = x
        logits = tf.matmul(Q, K, transpose_b=True) / self.scale

        if self.clip is not None:
            logits = self.clip * tf.math.tanh(logits)

        if self.return_logits:
            if mask is not None:
                logits = tf.where(tf.transpose(mask, perm=(0, 2, 1)), tf.ones_like(logits) * (-np.inf), logits)
            return logits

        if mask is not None:
            logits = tf.where(mask[:, None, None, :, 0], tf.ones_like(logits) * (-np.inf), logits)

        probs = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(probs, V)