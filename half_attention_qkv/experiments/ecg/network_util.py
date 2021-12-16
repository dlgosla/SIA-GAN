import copy
from typing import Optional, Any, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Dropout, Linear, LayerNorm, ModuleList
# from ..init import xavier_uniform_
from torch.nn.parameter import Parameter
import math

'''
class ArcFace(nn.Module):
    """
    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698
 
    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """
    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.5,
                 regularizer=None,
                 name='arcface',
                 **kwargs):
 
        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer
 
    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')
 
    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """
        embedding, label = inputs
 
        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')
 
        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')
 
        training = resolve_training_flag(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + self._m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output
 '''
            

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features #입력 차원
        self.out_features = out_features #클래스 개수
        self.s = s #scale
        self.m = m #margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)) #입력차원 x 클래스 개수
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # -- cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # normalization -> cosine
        
        # -- cos(theta + m)                
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1)) # sin(theta) = sqrt(1 - cos^2)
        phi = cosine * self.cos_m - sine * self.sin_m # cos(x+m) = cos(x)*cos(m) + sin(x) * sin(m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine) #cos이 0보다 크면 cos(theta+m) = cos(theta+m) 아니면 cos(theta+m) = cos(theta)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm) #cos(theta)가 cos(pi-m)보다크면 cos(theta+m) = cos(theta+m), 아니면 테일러 급수로 근사치 구함
        # --covert label to one hot
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        
        
        output = phi
        
        '''
        print("one got", one_hot.shape)
        print("one_hot", label.view(-1, 1).long().shape)        
        
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        
        #-- [0,0,0,1,0,0] -> [cos, cos, cos, phi, cos, cos]
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        '''
        
        # print(output)

        return output


class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttentionLayer(d_model, nhead, qkv_fc_layer=nn.Linear(d_model,d_model), fc_layer=nn.Linear(d_model,d_model))
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm((128,d_model), eps=layer_norm_eps)
        self.norm2 = LayerNorm((128,d_model), eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        #- src: [bs,256,50]

        #- self attention
        modal1 = src[:, :128, :]
        modal2 = src[:, 128:, :]

        src2 = self.self_attn(modal1, modal2, modal2, mask=src_mask)
        #- add & norm
        modal1 = modal1 + self.dropout1(src2)
        modal1 = self.norm1(modal1)

        #- FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(modal1))))
        #- add & norm
        modal1 = modal1 + self.dropout2(src2)
        modal1 = self.norm2(modal1)


       #- self attention
        

        src2 = self.self_attn(modal2, modal1, modal1, mask=src_mask)
        #- add & norm
        modal2 = modal2 + self.dropout1(src2)
        modal2 = self.norm1(modal2)

        #- FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(modal2))))
        #- add & norm
        modal2 = modal2 + self.dropout2(src2)
        modal2 = self.norm2(modal2)

        out = torch.cat([modal1, modal2], dim=1)
        return out


class MultiHeadAttentionLayer(Module):

	def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
		# qkv_fc_layer's shape: (d_embed, d_model)
		# fc_layer's shape: (d_model, d_embed)
		super(MultiHeadAttentionLayer, self).__init__()
		self.d_model = d_model
		self.h = h
		self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
		self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
		self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
		self.fc_layer = fc_layer


	def forward(self, query, key, value, mask=None): #[bs,256,50]
		# query, key, value's shape: (n_batch, seq_len, d_embed)
		# mask's shape: (n_batch, seq_len, seq_len)
		n_batch = query.shape[0] # get n_batch

		def transform(x, fc_layer): # reshape (n_batch, seq_len, d_embed) to (n_batch, h, seq_len, d_k)
			out = fc_layer(x) # out's shape: (n_batch, seq_len, d_model)
			out = out.view(n_batch, -1, self.h, self.d_model//self.h) # out's shape: (n_batch, seq_len, h, d_k)
			out = out.transpose(1, 2) # out's shape: (n_batch, h, seq_len, d_k)
			return out

		query = transform(query, self.query_fc_layer) # query, key, value's shape: (n_batch, h, seq_len ,d_k)
		key = transform(key, self.key_fc_layer)
		value = transform(value, self.value_fc_layer)

		if mask is not None:
			mask = mask.unsqueeze(1) # mask's shape: (n_batch, 1, seq_len, seq_len)

		out = calculate_attention(query, key, value, mask) # out's shape: (n_batch, h, seq_len, d_k)
		out = out.transpose(1, 2) # out's shape: (n_batch, seq_len, h, d_k)
		out = out.contiguous().view(n_batch, -1, self.d_model) # out's shape: (n_batch, seq_len, d_model)
		out = self.fc_layer(out) # out's shape: (n_batch, seq_len, d_embed)
		return out


def calculate_attention(query, key, value, mask):
    # query, key, value's shape: (n_batch, seq_len, d_k)
    d_k = key.size(-1) # get d_k
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, attention_score's shape: (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k) # scaling
    if mask is not None:
        attention_score = score.masked_fill(mask==0, -1e9) # masking
    attention_prob = F.softmax(attention_score, dim=-1) # softmax, attention_prob's shape: (n_batch, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # Attention_Prob x V, out's shape: (n_batch, seq_len, d_k)
    return out


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))