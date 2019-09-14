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
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)
    self.final_linear = nn.Linear(model_dim, model_dim)

  def forward(self, key, value, query, mask=None,
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

    # CHECKS
    # batch, k_len, d = key.size()
    # batch_, k_len_, d_ = value.size()
    # aeq(batch, batch_)
    # aeq(k_len, k_len_)
    # aeq(d, d_)
    # batch_, q_len, d_ = query.size()
    # aeq(batch, batch_)
    # aeq(d, d_)
    # aeq(self.model_dim % 8, 0)
    # if mask is not None:
    #    batch_, q_len_, k_len_ = mask.size()
    #    aeq(batch_, batch)
    #    aeq(k_len_, k_len)
    #    aeq(q_len_ == q_len)
    # END CHECKS

    batch_size = key.size(0)
    dim_per_head = self.dim_per_head
    head_count = self.head_count
    key_len = key.size(1)
    query_len = query.size(1)

    def shape(x):
      """  projection """
      return x.view(batch_size, -1, head_count, dim_per_head) \
          .transpose(1, 2)

    def unshape(x):
      """  compute context """
      return x.transpose(1, 2).contiguous() \
              .view(batch_size, -1, head_count * dim_per_head)

    # 1) Project key, value, and query.
    if layer_cache is not None:
      if type == "self":
          query, key, value = self.linear_query(query),\
                              self.linear_keys(query),\
                              self.linear_values(query)

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
      key = shape(key)
      value = shape(value)

    query = shape(query)

    key_len = key.size(2)
    query_len = query.size(2)

    # 2) Calculate and scale scores.
    query = query / math.sqrt(dim_per_head)
    scores = torch.matmul(query, key.transpose(2, 3))

    if mask is not None:
        mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
        scores = scores.masked_fill(mask, -1e18)

    # 3) Apply attention dropout and compute context vectors.
    attn = self.softmax(scores)
    drop_attn = self.dropout(attn)
    context = unshape(torch.matmul(drop_attn, value))

    output = self.final_linear(context)
    # CHECK
    # batch_, q_len_, d_ = output.size()
    # aeq(q_len, q_len_)
    # aeq(batch, batch_)
    # aeq(d, d_)

    # Return one attn
    top_attn = attn \
        .view(batch_size, head_count,
              query_len, key_len)[:, 0, :, :] \
        .contiguous()

    return output, top_attn

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

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
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x
