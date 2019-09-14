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
    self.structure_attn = onmt.sublayer.MultiHeadedAttention(heads, 64, dropout=dropout)
    self.structure_forward = onmt.sublayer.StructureFeedForward(128, dropout=dropout)   # d_a
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)   #d_ff (int): the hidden layer size of the second-layer of the FNN.
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)   #FeedForwardnorm
    self.structure_layer_norm = nn.LayerNorm(64, eps=1e-6)
    self.dropout = nn.Dropout(dropout)
    self.cnn = nn.Conv1d(64, 64, 4)


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

    structure = structure.transpose(0, 1).contiguous()
    structure = structure.transpose(0, 2).contiguous()     # structuer  146 * 144 * 5 * 64
    batch_size = structure.size(0)
    edge_size = structure.size(1) ** 0.5
    edge_size = int(edge_size)
    structure = structure.view(-1, 4, 64).contiguous()
    structure, _ = self.structure_attn(structure, structure, structure)      # -1 * 5 * 64

    # attn = self.structure_forward(structure)      # -1 * 1 * 5
    # output_structures = torch.matmul(attn, structure)           # -1 * 1 * 64
    structure = structure.transpose(1, 2)   # -1, 64, 5
    structure = self.cnn(structure)
    structure = torch.relu(structure)
    structure = self.dropout(structure)
    output_structures = structure    

    output_structures = output_structures.view(batch_size, edge_size, edge_size, 64)
    output_structures = self.structure_layer_norm(output_structures)
    outputs, _ = self.self_attn(input_norm, input_norm, input_norm, structure=output_structures, mask=mask)   # structure 146 * 12 * 12 * 64
    inputs = self.dropout(outputs) + inputs
    input_norm = self.ffn_layer_norm(inputs)
    outputs = self.feed_forward(input_norm)
    inputs = outputs + inputs
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

    structure_emb1 = self.structure_embeddings(structure1, 0)              # 12 * 12 * batch * embedding_dim(64)
    structure_emb2 = self.structure_embeddings(structure2, 1)
    structure_emb3 = self.structure_embeddings(structure3, 2)
    structure_emb4 = self.structure_embeddings(structure4, 3)
    # structure_emb6 = self.structure_embeddings(structure6, 5)
    # structure_emb7 = self.structure_embeddings(structure7, 6)
    # structure_emb8 = self.structure_embeddings(structure8, 7)


    batch_size = structure_emb1.size(2)  # batch的大小

    structure_emb1 = structure_emb1.view(-1, 1, batch_size, 64).contiguous()       # -1 * 1 * batch * 64        [144, 1, 146, 64]
    structure_emb2 = structure_emb2.view(-1, 1, batch_size, 64).contiguous()
    structure_emb3 = structure_emb3.view(-1, 1, batch_size, 64).contiguous()
    structure_emb4 = structure_emb4.view(-1, 1, batch_size, 64).contiguous()
    # structure_emb6 = structure_emb6.view(-1, 1, batch_size, 64).contiguous()
    # structure_emb7 = structure_emb7.view(-1, 1, batch_size, 64).contiguous()
    # structure_emb8 = structure_emb8.view(-1, 1, batch_size, 64).contiguous()
    # 144 * 5 * 146 * 64
    output_structure = torch.cat((structure_emb1, structure_emb2, structure_emb3, structure_emb4), 1)

    out = emb.transpose(0, 1).contiguous()          # 146 * 12 * 512

    # output_structure1 = structure_emb1.transpose(0, 1).contiguous()
    # output_structure2 = structure_emb2.transpose(0, 1).contiguous()
    # output_structure3 = structure_emb3.transpose(0, 1).contiguous()
    # output_structure4 = structure_emb4.transpose(0, 1).contiguous()
    # output_structure5 = structure_emb5.transpose(0, 1).contiguous()
    # output_structure6 = structure_emb6.transpose(0, 1).contiguous()
    # output_structure7 = structure_emb7.transpose(0, 1).contiguous()
    # output_structure8 = structure_emb8.transpose(0, 1).contiguous()
    # output_structure = output_structure1 + output_structure2 + output_structure3 + output_structure4 + output_structure5

    words = src.transpose(0, 1)
    padding_idx = self.embeddings.word_padding_idx
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    # Run the forward pass of every layer of the tranformer.
    for i in range(self.num_layers):
      out = self.transformer[i](out, output_structure, mask)
    out = self.layer_norm(out)

    return emb, out.transpose(0, 1).contiguous(), lengths

