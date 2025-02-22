import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
import numpy as np


class Propagation_Attention(layers.Layer):

    def __init__(self,
                 dim,
                 bias_table_size=10,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 name=None,
                 mask_rate=0.5):
        super(Propagation_Attention, self).__init__(name=name)
        self.bias_table_size = bias_table_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv",
                                kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim, name="out",
                                 kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())
        self.proj_drop = layers.Dropout(proj_drop_ratio)
        self.mask_rate = mask_rate

        self.relative_position_bias_table = self.add_weight(
            shape=[self.bias_table_size, self.num_heads],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32,
            name=f"relative_position_bias_table{mask_rate}"
        )

    def mask_matrix(self, path_len_li, relation_position):
        relation_position_li = relation_position.numpy()
        rel_mask_li = []
        for index in range(len(path_len_li)):
            path_len = path_len_li[index]
            relation_position_temp = relation_position_li[index]
            mask_len = int(int(path_len) * self.mask_rate)
            mask_table = np.zeros([self.bias_table_size, self.num_heads], dtype=np.float32)
            for i in range(mask_len + 1, self.bias_table_size):
                for j in range(self.num_heads):
                    mask_table[i][j] = -100
            rel_mask_table = tf.gather(mask_table, relation_position_temp)
            rel_mask_li.append(rel_mask_table)
        rel_mask_li = np.asarray(rel_mask_li, dtype=np.float32)
        rel_mask_li = tf.cast(rel_mask_li, dtype=tf.float32)
        return rel_mask_li

    def call(self, inputs, training=None):
        (att_embedding, relation_position, rel_len) = inputs
        relative_mask_bias = self.mask_matrix(path_len_li=rel_len, relation_position=relation_position)
        relative_mask_bias = tf.transpose(relative_mask_bias, [0, 3, 1, 2])

        relative_position_bias = tf.gather(self.relative_position_bias_table, relation_position)
        relative_position_bias = tf.transpose(relative_position_bias, [0, 3, 1, 2])

        B, N, C = att_embedding.shape

        qkv = self.qkv(att_embedding)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = attn + relative_position_bias + relative_mask_bias

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x, relation_position, rel_len


class MLP(layers.Layer):

    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(in_features, name="Dense_1",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class Block(layers.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 mlp_drop_ratio=0.,
                 attn_drop_ratio=0.,
                 block_drop_ratio=0.,
                 mask_rate=1,
                 name=None):
        super(Block, self).__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.variable_windows_attn = Propagation_Attention(dim, num_heads=num_heads,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale, mask_rate=mask_rate,
                                                           attn_drop_ratio=attn_drop_ratio,
                                                           proj_drop_ratio=mlp_drop_ratio,
                                                           name=f"VW-MSA-mask_rate: {mask_rate}")
        self.block_drop = layers.Dropout(block_drop_ratio)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = MLP(dim, drop=mlp_drop_ratio, name="MlpBlock")

    def call(self, inputs, training=None):
        (att_embedding, relation_position, rel_len) = inputs
        input_att_embedding = att_embedding
        att_embedding = self.norm1(att_embedding)
        (att_embedding, relation_position, rel_len) = self.variable_windows_attn(
            (att_embedding, relation_position, rel_len))
        x = input_att_embedding + self.block_drop(att_embedding, training=training)
        x = x + self.block_drop(self.mlp(self.norm2(x)), training=training)
        return (x, relation_position, rel_len)


class Propagation_Transformer(Model):
    def __init__(self, embed_dim=768,
                 depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                 mlp_drop_ratio=0., attn_drop_ratio=0., block_drop_ratio=0.,
                 num_classes=1000, mask_rate_li=[], name=''):
        super(Propagation_Transformer, self).__init__(name=name)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.qkv_bias = qkv_bias

        self.pos_drop = layers.Dropout(mlp_drop_ratio)

        self.blocks = []
        for i in range(len(mask_rate_li)):
            mask_rate = mask_rate_li[i]
            block = Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, mask_rate=mask_rate,
                          qk_scale=qk_scale, mlp_drop_ratio=mlp_drop_ratio, attn_drop_ratio=attn_drop_ratio,
                          block_drop_ratio=block_drop_ratio, name=f'EncoderBlock{i}')
            self.blocks.append(block)

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.head = layers.Dense(num_classes, name="head", kernel_initializer=initializers.GlorotUniform())

        self.p = self.add_weight(
            shape=[4, 1],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32,
            name="p"
        )

    def call(self, inputs, training=None):
        (att_embedding, relation_position, rel_len, att_len) = inputs
        att_embedding = att_embedding @ self.p
        att_embedding = tf.squeeze(att_embedding, axis=-1)
        att_embedding = self.pos_drop(att_embedding, training=training)
        x = (att_embedding, relation_position, rel_len)

        for block in self.blocks:
            x = block(x, training=training)

        (att_embedding, relation_position, rel_len) = x
        att_embedding = tf.squeeze(att_embedding)
        att_embedding = self.norm(att_embedding)

        output = self.head(att_embedding)
        return output
