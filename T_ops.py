import tensorflow as tf

from keras import backend as K
from keras.layers import Lambda
from keras import initializers
from keras.engine import InputSpec, Layer


def pairwise_dis(vests):
    x, y = vests
    return K.abs(x-y)

def pairwise_mul(vests):
    x, y = vests
    return x*y

def pairwise_sum(vests):
    x, y = vests
    return x+y

def last_timestep(vests):
    return vests[:,-1,:]

def first_timestep(vests):
    return vests[:,0,:]

def last_timestep_slice(vests):
    rdim = int(K.int_shape(vests)[-1]/2)
    return vests[:,-1,:rdim]

def first_timestep_slice(vests):
    rdim = int(K.int_shape(vests)[-1]/2)
    return vests[:,0,rdim:]

def sum_over_time(vests):
    return K.sum(vests, axis=-2)

def pairwise_mulsq(vests):
    x, y = vests
    return K.square(x*y)

def cosine_similarity(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum((x * y), axis=-1, keepdims=True)

def cosine_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

def leaky_relu(inputs, alpha=0.1):
    return K.relu(inputs, alpha=alpha)

def weighted_sum(vests):
    data, weights = vests
    return K.sum(data*weights, axis=-2)

def mysoftmax(x, axis=-1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
def top_kmax(x):
    x=tf.transpose(x, [0, 2, 1])
    k_max = tf.nn.top_k(x, k=top_k)
    return tf.reshape(k_max[0], (-1, num_filters[-1]*top_k))
        
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
        
class SoftAlignment(object):
    def __init__(self):
        pass

    def __call__(self, sentence, attention, transpose=False):
        
        def apply_attention(attmat):
            att = attmat[0]
            mat = attmat[1]
            if transpose:
                att = K.permute_dimensions(att,(0, 2, 1))
            return K.batch_dot(att, mat)
        
        return Lambda(apply_attention)([attention, sentence])