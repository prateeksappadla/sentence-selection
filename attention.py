"""
Goal: Create classes to easily implement different attention mechanisms.

Issues:
    - MUST MAKE SOFTMAX NUMERICALLY STABLE
    - Find out what initializer to user for bilinear weight

Credits: Idea from https://arxiv.org/pdf/1606.02858v2.pdf
"""

import tensorflow as tf

# class BilinearFunction(object):
#     def __init__(self, attending_size, attended_size, scope=None, W_bilinear_const=None):
#       self._attending_size = attending_size
#       self._attended_size = attended_size
#       self._scope = scope
#       self.W_bilinear_const = W_bilinear_const
#
#     # Expect dimensions: attending (batch x attending_size),
#         # attended (batch x time x attended_size) - time could be other dim value
#     def __call__(self, attending, attended, scope=None):
#       with tf.variable_scope(self._scope or type(self).__name__):  # "BilinearFunction"
#           attending_size = self._attending_size
#           attended_size= self._attended_size
#
#           batch_size = tf.shape(attended)[0]
#
#           # different initializer?
#           if self.W_bilinear_const != None:
#               W_bilinear = self.W_bilinear_const
#               batch_size = 1
#               attended = tf.reshape(attended, [batch_size, -1, attended_size])
#           else:
#               W_bilinear = tf.get_variable(name="bilinear_attention", shape=[attending_size, attended_size], \
#                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
#
#           # Dimensions (batch x attended_size)
#           # attendded: (batch x time x attended_size)
#           attending_tensor = tf.matmul(attending, W_bilinear)
#           attending_tensor = tf.reshape(attending_tensor, [batch_size, attended_size, 1])
#
#           # Now take dot products of attending tensor with each timestep of attended tensor
#           # Should return matrix of attention weights with dimensions (batch x time)
#
#           # multiplies each slice with each other respective slice - EXPLAIN BETTER
#           dot_prod = tf.batch_matmul(attended, attending_tensor)
#
#           # Should return matrix of attention weights with dimensions (batch x time)
#           dot_prod = tf.squeeze(dot_prod, [2])
#
#           # Custom Softmax b/c need to use time_mask --------------------
#           # Also numerical stability: alpha_weights = tf.nn.softmax(dot_prod)
#           #max(x, 0) - x * z + log(1 + exp(-abs(x)))
#
#           #max_vals = tf.mul(tf.ones_like(dot_prod), tf.expand_dims(tf.reduce_max(dot_prod, 1), -1))
#           numerator = tf.exp(tf.sub(dot_prod, tf.expand_dims(tf.reduce_max(dot_prod, 1), -1))) #* seq_len_mask
#           denom = tf.reduce_sum(numerator, 1)
#
#           # Transpose so broadcasting scalar division works properly
#           # Dimensions (batch x time)
#           alpha_weights = tf.transpose(tf.div(tf.transpose(numerator), denom), name="alpha_weights")
#
#           # Find weighted sum of attended tensor using alpha_weights
#           # attended dimensions: (batch x time x attended_size)
#           tf.mul(attended, alpha_weights)
#
#           # Again must permute axes so broadcasting scalar multiplication works properly
#           attended_transposed = tf.transpose(attended, perm=[2,0,1])
#           attended_weighted_transposed = tf.mul(attended_transposed, alpha_weights)
#           attended_weighted = tf.transpose(attended_weighted_transposed, perm=[1,2,0])
#           # attend_result dimensions (batch x attended_size)
#           attend_result = tf.reduce_sum(attended_weighted, 1, name="attend_result")
#
#           return (alpha_weights, attend_result)

class BilinearFunction(object):
    def __init__(self, attending_size, attended_size, scope=None):
      self._attending_size = attending_size
      self._attended_size = attended_size
      self._scope = scope

    # Expect dimensions: attending (batch x attending_size),
        # attended (batch x time x attended_size) - time could be other dim value
    def __call__(self, attending, attended, time_mask, scope=None):
      with tf.variable_scope(self._scope or type(self).__name__):  # "BilinearFunction"
          attending_size = self._attending_size
          attended_size= self._attended_size

          batch_size = tf.shape(attended)[0]

          # different initializer?
          self.W_bilinear = tf.get_variable(name="bilinear_attention", shape=[attending_size, attended_size], \
              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1), dtype=tf.float32)

          # Dimensions (batch x attended_size)
          attending_tensor = tf.matmul(attending, self.W_bilinear)
          attending_tensor = tf.reshape(attending_tensor, [batch_size, attended_size, 1])

          # Now take dot products of attending tensor with each timestep of attended tensor
          # Should return matrix of attention weights with dimensions (batch x time)

          # multiplies each slice with each other respective slice - EXPLAIN BETTER
          dot_prod = tf.squeeze(tf.batch_matmul(attended, attending_tensor)) * time_mask

          # Custom Softmax b/c need to use time_mask --------------------
          # Also numerical stability: alpha_weights = tf.nn.softmax(dot_prod)

          # e_x = exp(x - x.max(axis=1))
          # out = e_x / e_x.sum(axis=1)
          numerator = tf.exp(tf.sub(dot_prod, tf.expand_dims(tf.reduce_max(dot_prod, 1), -1))) * time_mask
          denom = tf.reduce_sum(numerator, 1) # custom gather, then reduce sum

          # Dimensions (batch x time)
          alpha_weights = tf.div(numerator, tf.expand_dims(denom, 1))

          # Find weighted sum of attended tensor using alpha_weights
          # attended dimensions: (batch x time x attended_size)
          attended_weighted = tf.mul(attended, tf.expand_dims(alpha_weights, -1))

          # attend_result dimensions (batch x attended_size)
          attend_result = tf.reduce_sum(attended_weighted, 1)

          return (alpha_weights, attend_result)
