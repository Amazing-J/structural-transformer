""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
  """
  Multi-Head Attention module from
  "Attention is All You Need"
  :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

  Similar to standard `dot` attention but uses
  multiple attention distributions simulataneously
  to select relevant items.

  Args:
     head_count (int): number of parallel heads
     model_dim (int): the dimension of keys/values/queries,
         must be divisible by head_count
     dropout (float): dropout parameter
  """

  def __init__(self, head_count, model_dim, dropout=0.1):
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim

    super(MultiHeadedAttention, self).__init__()
    self.head_count = head_count

    self.linear_keys = nn.Linear(model_dim,
                                 head_count * self.dim_per_head)
    self.linear_values = nn.Linear(model_dim,
                                   head_count * self.dim_per_head)
    self.linear_query = nn.Linear(model_dim,
                                  head_count * self.dim_per_head)
    self.linear_structure_k = nn.Linear(64, 64)
    self.linear_structure_v = nn.Linear(64, 64)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)
    self.final_linear = nn.Linear(model_dim, model_dim)

  def forward(self, key, value, query, structure=None, mask=None,
              layer_cache=None, type=None):
    """
    Compute the context vector and the attention vectors.

    Args:
       key (`FloatTensor`): set of `key_len`
            key vectors `[batch, key_len, dim]`
       value (`FloatTensor`): set of `key_len`
            value vectors `[batch, key_len, dim]`
       query (`FloatTensor`): set of `query_len`
             query vectors  `[batch, query_len, dim]`
       mask: binary mask indicating which keys have
             non-zero attention `[batch, query_len, key_len]`
    Returns:
       (`FloatTensor`, `FloatTensor`) :

       * output context vectors `[batch, query_len, dim]`
       * one of the attention vectors `[batch, query_len, key_len]`
    """
    global structure_k, structure_v

    batch_size = key.size(0)
    dim_per_head = self.dim_per_head
    head_count = self.head_count
    key_len = key.size(1)
    query_len = query.size(1)

    def shape(x):
      """  projection """
      return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

    def unshape(x):
      """  compute context """
      return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

    # 1) Project key, value, and query.
    if layer_cache is not None:
      if type == "self":
          query, key, value = self.linear_query(query),\
                              self.linear_keys(query),\
                              self.linear_values(query)

          if structure is not None:
                structure_k, structure_v = self.linear_structure_k(structure),\
                                           self.linear_structure_v(structure)
          else:
                structure_k = None
                structure_v = None
          key = shape(key)
          value = shape(value)

          if layer_cache is not None:
              device = key.device
              if layer_cache["self_keys"] is not None:
                  key = torch.cat(
                      (layer_cache["self_keys"].to(device), key),
                      dim=2)
              if layer_cache["self_values"] is not None:
                  value = torch.cat(
                      (layer_cache["self_values"].to(device), value),
                      dim=2)
              layer_cache["self_keys"] = key
              layer_cache["self_values"] = value
      elif type == "context":
        query = self.linear_query(query)
        if layer_cache is not None:
          if layer_cache["memory_keys"] is None:
            key, value = self.linear_keys(key),\
                         self.linear_values(value)
            key = shape(key)
            value = shape(value)
          else:
            key, value = layer_cache["memory_keys"],\
                       layer_cache["memory_values"]
          layer_cache["memory_keys"] = key
          layer_cache["memory_values"] = value
        else:
          key, value = self.linear_keys(key),\
                       self.linear_values(value)
          key = shape(key)
          value = shape(value)
    else:
      key = self.linear_keys(key)
      value = self.linear_values(value)
      query = self.linear_query(query)

      if structure is not None:
                 structure_k, structure_v = self.linear_structure_k(structure),\
                                            self.linear_structure_v(structure)
      else:
                 structure_k = None
                 structure_v = None
      key = shape(key)
      value = shape(value)

    query = shape(query)

    key_len = key.size(2)
    query_len = query.size(2)

    # 2) Calculate and scale scores.
    query = query / math.sqrt(dim_per_head)
    scores = torch.matmul(query, key.transpose(2, 3))

    if structure_k is not None:
        q = query.transpose(1,2)

        #print(q.size(), structure_k.transpose(2,3).size())
        scores_k = torch.matmul(q, structure_k.transpose(2,3))
        scores_k = scores_k.transpose(1,2)
        #print (scores.size(),scores_k.size())
        scores = scores + scores_k
    if mask is not None:
        mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
        scores = scores.masked_fill(mask, -1e18)

    # 3) Apply attention dropout and compute context vectors.
    attn = self.softmax(scores)
    drop_attn = self.dropout(attn)
    context = torch.matmul(drop_attn, value)
    if structure_v is not None:
        drop_attn_v = drop_attn.transpose(1,2)
        context_v = torch.matmul(drop_attn_v, structure_v)
        context_v = context_v.transpose(1,2)
        #print(context.size(),context_v.size())
        context = context + context_v
    context = unshape(context)
    output = self.final_linear(context)

    # Return one attn
    top_attn = attn \
        .view(batch_size, head_count,
              query_len, key_len)[:, 0, :, :] \
        .contiguous()

    return output, top_attn

class PositionwiseFeedForward(nn.Module):
  """ A two-layer Feed-Forward-Network.

      Args:
          d_model (int): the size of input for the first-layer of the FFN.
          d_ff (int): the hidden layer size of the second-layer
                            of the FNN.
          dropout (float): dropout probability(0-1.0).
  """

  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.relu = nn.ReLU()

  def forward(self, x):
    """
    Layer definition.

    Args:
        input: [ batch_size, input_len, model_dim ]


    Returns:
        output: [ batch_size, input_len, model_dim ]
    """
    inter = self.dropout_1(self.relu(self.w_1(x)))
    output = self.w_2(inter)
    return output


class StructureFeedForward(nn.Module):

  def __init__(self, d_a, dropout=0.1):
    super(StructureFeedForward, self).__init__()
    self.w_s1 = nn.Linear(64, d_a)
    self.w_s2 = nn.Linear(d_a, 1)
    self.tanh = nn.Tanh()
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):                                  # [ structure_size, 5, 64 ]  ->  [structure_size, 1, 5]
      x = self.dropout(self.tanh(self.w_s1(x)))          # [size, 5, d_a]
      x = self.w_s2(x)                                   # [size, 5, 1]
      x = x.transpose(1, 2).contiguous()                 # [size, 1, 5]
      x = self.softmax(x)
      return x

