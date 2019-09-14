"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward
import torch
import math


class TransformerEncoderLayer(nn.Module):
  """
      A single layer of the transformer encoder.
      Args:
          d_model (int): the dimension of keys/values/queries in
                     MultiHeadedAttention, also the input size of
                     the first-layer of the PositionwiseFeedForward.
          heads (int): the number of head for MultiHeadedAttention.
          d_ff (int): the second-layer of the PositionwiseFeedForward.
          dropout (float): dropout probability(0-1.0).
  """
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerEncoderLayer, self).__init__()
    self.self_attn = onmt.sublayer.MultiHeadedAttention(heads, d_model, dropout=dropout)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)   #d_ff (int): the hidden layer size of the second-layer of the FNN.
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)   #FeedForwardnorm
    self.structure_layer_norm = nn.LayerNorm(64, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

    # self.weight_1 = torch.nn.Parameter(torch.Tensor(), requires_grad=True)   # 增加可学习参数
    # self.weight_1.data.fill_(0.25)

  def forward(self, inputs, structure, mask):
    """
           Transformer Encoder Layer definition.

           Args:
               inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`  146 * 12 * 512
               mask (`LongTensor`): `[batch_size x src_len x src_len]`
               structure: 144 * 5 * 146 * 64
           Returns:
               (`FloatTensor`):

               * outputs `[batch_size * src_len * model_dim]`
    """
    input_norm = self.att_layer_norm(inputs)

    structure = structure.transpose(1, 2)
    structure = structure.transpose(0, 1)
    structure = self.structure_layer_norm(structure)

    outputs, _ = self.self_attn(input_norm, input_norm, input_norm, structure=structure, mask=mask)

    inputs = self.dropout(outputs) + inputs
    input_norm = self.ffn_layer_norm(inputs)
    outputs = self.feed_forward(input_norm)
    inputs = outputs +inputs
    return inputs




class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, structure_embeddings):
    super(TransformerEncoder, self).__init__()

    self.num_layers = num_layers
    self.embeddings = embeddings
    self.structure_embeddings = structure_embeddings

    #Bulid Encode
    self.transformer = nn.ModuleList( [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def _check_args(self, src, lengths=None):
    _, n_batch = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)

  def forward(self, src, structure1, structure2, structure3, structure4, structure5, lengths=None):
    """ See :obj:`EncoderBase.forward()`"""
    #self._check_args(src, lengths)

    emb = self.embeddings(src)
    assert emb.dim() == 3                  # len * batch * embedding_dim

    structure_emb1 = self.structure_embeddings(structure1)              # 12 * 12 * batch * embedding_dim(64)
    structure_emb2 = self.structure_embeddings(structure2)
    structure_emb3 = self.structure_embeddings(structure3)
    structure_emb4 = self.structure_embeddings(structure4)
    structure_emb5 = self.structure_embeddings(structure5)
    # structure_emb6 = self.structure_embeddings(structure6, 5)
    # structure_emb7 = self.structure_embeddings(structure7, 6)
    # structure_emb8 = self.structure_embeddings(structure8, 7)
    assert structure_emb1.dim() == 4 and structure_emb2.dim() == 4 and structure_emb3.dim() == 4 and structure_emb4.dim() == 4 




    out = emb.transpose(0, 1).contiguous()          # 146 * 12 * 512

    output_structure1 = structure_emb1.transpose(0, 1).contiguous()
    output_structure2 = structure_emb2.transpose(0, 1).contiguous()
    output_structure3 = structure_emb3.transpose(0, 1).contiguous()
    output_structure4 = structure_emb4.transpose(0, 1).contiguous()
    output_structure5 = structure_emb5.transpose(0, 1).contiguous()
    # output_structure8 = structure_emb8.transpose(0, 1).contiguous()
    output_structure = (output_structure1 + output_structure2 + output_structure3 + output_structure4 + output_structure5) * (1/5)

    words = src.transpose(0, 1)
    padding_idx = self.embeddings.word_padding_idx
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    # Run the forward pass of every layer of the tranformer.
    for i in range(self.num_layers):
      out = self.transformer[i](out, output_structure, mask)
    out = self.layer_norm(out)

    return emb, out.transpose(0, 1).contiguous(), lengths

