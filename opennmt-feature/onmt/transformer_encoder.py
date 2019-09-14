"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward



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

  def forward(self, inputs, structure, mask):
    """
           Transformer Encoder Layer definition.

           Args:
               inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
               mask (`LongTensor`): `[batch_size x src_len x src_len]`

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

  def forward(self, src, structure, lengths=None):
    """ See :obj:`EncoderBase.forward()`"""
    #self._check_args(src, lengths)

    emb = self.embeddings(src)

    assert emb.dim() == 3 # len x batch x embedding_dim
    structure_emb = self.structure_embeddings(structure)
    assert structure_emb.dim() == 4

    out = emb.transpose(0, 1).contiguous()
    output_structure = structure_emb.transpose(0,1).contiguous()

    #print(output_structure.size())

    words = src.transpose(0, 1)
    padding_idx = self.embeddings.word_padding_idx
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    # Run the forward pass of every layer of the tranformer.
    for i in range(self.num_layers):
      out = self.transformer[i](out, output_structure, mask)
    out = self.layer_norm(out)

    return emb, out.transpose(0, 1).contiguous(), lengths

