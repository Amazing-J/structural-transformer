"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.sublayer import PositionwiseFeedForward

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerDecoderLayer, self).__init__()

    self.self_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)

    self.context_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
    self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
    
    self.dropout = dropout
    self.drop = nn.Dropout(dropout)
    mask = self._get_attn_subsequent_mask(MAX_SIZE)
    # Register self.mask as a buffer in TransformerDecoderLayer, so
    # it gets TransformerDecoderLayer's cuda behavior automatically.
    self.register_buffer('mask', mask)

  def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
              layer_cache=None, step=None):
    dec_mask = None
    if step is None:
      dec_mask = torch.gt(tgt_pad_mask +
                          self.mask[:, :tgt_pad_mask.size(-1),
                                    :tgt_pad_mask.size(-1)], 0)

    input_norm = self.layer_norm_1(inputs)

    query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=dec_mask,
                                 layer_cache=layer_cache,
                                 type="self")

    query = self.drop(query) + inputs

    query_norm = self.layer_norm_2(query)
    mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                  mask=src_pad_mask,
                                  layer_cache=layer_cache,
                                  type="context")
    output = self.feed_forward(self.drop(mid) + query)

    return output, attn

  def _get_attn_subsequent_mask(self, size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
    super(TransformerDecoder, self).__init__()

    # Basic attributes.
    self.decoder_type = 'transformer'
    self.num_layers = num_layers
    self.embeddings = embeddings

    # Decoder State
    self.state = {}

    # Build TransformerDecoder.
    self.transformer_layers = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def init_state(self, src, src_enc):
    """ Init decoder state """
    self.state["src"] = src
    self.state["src_enc"] = src_enc
    self.state["cache"] = None

  def map_state(self, fn):
    def _recursive_map(struct, batch_dim=0):
      for k, v in struct.items():
        if v is not None:
          if isinstance(v, dict):
            _recursive_map(v)
          else:
            struct[k] = fn(v, batch_dim)

    self.state["src"] = fn(self.state["src"], 1)
    self.state["src_enc"] = fn(self.state["src_enc"], 1)
    if self.state["cache"] is not None:
      _recursive_map(self.state["cache"])

  def detach_state(self):
    self.state["src"] = self.state["src"].detach()

  def forward(self, tgt, step=None):
    """
    See :obj:`onmt.modules.RNNDecoderBase.forward()`
    """
    if step == 0:
      self._init_cache(self.num_layers)

    src = self.state["src"]
    memory_bank = self.state["src_enc"]
    src_words = src.transpose(0, 1)
    tgt_words = tgt.transpose(0, 1)

    # Initialize return variables.
    attns = {"std": []}

    # Run the forward pass of the TransformerDecoder.
    emb = self.embeddings(tgt, step=step)
    assert emb.dim() == 3  # len x batch x embedding_dim

    output = emb.transpose(0, 1).contiguous()
    src_memory_bank = memory_bank.transpose(0, 1).contiguous()

    pad_idx = self.embeddings.word_padding_idx
    src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]
    tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

    for i in range(self.num_layers):
      output, attn = self.transformer_layers[i](
        output,
        src_memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=(
          self.state["cache"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)

    output = self.layer_norm(output)

    # Process the result and update the attentions.
    dec_outs = output.transpose(0, 1).contiguous()
    attn = attn.transpose(0, 1).contiguous()

    attns["std"] = attn

    # TODO change the way attns is returned dict => list or tuple (onnx)
    return dec_outs, attns

  def _init_cache(self, num_layers):
    self.state["cache"] = {}

    for l in range(num_layers):
      layer_cache = {
        "memory_keys": None,
        "memory_values": None
      }
      layer_cache["self_keys"] = None
      layer_cache["self_values"] = None
      self.state["cache"]["layer_{}".format(l)] = layer_cache
