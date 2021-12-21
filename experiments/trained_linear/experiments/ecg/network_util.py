import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, MultiheadAttention, Dropout, Linear, LayerNorm, ModuleList
# from ..init import xavier_uniform_


class TransformerEncoderLayer(Module):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def forward(self, src_Q: Tensor,src_K: Tensor,src_V: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        
        src2 = self.self_attn(src_Q, src_K, src_V, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src_Q + self.dropout1(src2)

        src = src.permute(1,0,2) #[bs,128,50]
        src = self.norm1(src)
        src = src.permute(1,0,2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        src = src.permute(1,0,2)
        src = self.norm2(src)
        src = src.permute(1,0,2)
        return src

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



# class TransformerEncoder(Module):
#     """TransformerEncoder is a stack of N encoder layers

#     Args:
#         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#         num_layers: the number of sub-encoder-layers in the encoder (required).
#         norm: the layer normalization component (optional).

#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = transformer_encoder(src)
#     """
#     __constants__ = ['norm']

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src_Q: Tensor, src_K: Tensor, src_V: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         """Pass the input through the encoder layers in turn.

#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         output_Q, output_K, output_V = src_Q, src_K, src_V

#         for mod in self.layers:
#             output = mod(output_Q, output_K, output_V, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
#         if self.norm is not None:
#             output = self.norm(output)

#         return output

# def _get_clones(module, N):
#     return ModuleList([copy.deepcopy(module) for i in range(N)])