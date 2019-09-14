""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import math
import onmt.constants as Constants

class Beam():
  ''' Beam Search '''
  def __init__(self, size, decode_length, bos_id, eos_id, minimal_length=0, alpha=0.6, stop_early=True, device=False):
    self.size = size
    self.alpha = alpha
    self.stop_early = stop_early
    self.decode_length = decode_length
    self.minimal_length = minimal_length
    self._done = False
    self.device = device
    self.minimal_score = -1.0 * 1e4
    self.eos_id = eos_id
    
    self.alive_seq = torch.zeros((size, 1), dtype=torch.long, device=device)
    self.alive_seq[0][0] = bos_id
    # alive_seq: (size, 1)
    self.alive_log_prob = torch.zeros((size,), dtype=torch.float, device=device)
    # alive_log_prob: (size, )
    
    # The score for each finished translation on the beam
    self.finished_seq = torch.zeros(self.alive_seq.size(), dtype=torch.long, device=device) + self.eos_id
    # finished_seq: (size, 1)
    self.finished_scores = torch.ones((size,), dtype=torch.float, device=device) * self.minimal_score
    # finished_scores: (size, 1)
    self.finished_flags = self.finished_scores > 0
    # finished_flags: (size, 1)
  
  def is_finished(self):
    if not self.stop_early:
      return self.alive_seq.size(1) < self.decode_length
    
    max_length_penalty = math.pow((5. + self.decode_length) / 6., self.alpha)
    lower_bound_alive_scores = self.alive_log_prob[0] / max_length_penalty
    # lower_bound_alive_scores: scalar
    lowest_score_of_fininshed_in_finished = torch.min(self.finished_scores * self.finished_flags.type(torch.float))
    # non-zero value (must be less than 0) if at least one hypothesis is finished, 
    # 0 if all hypothesis are not finished
    at_least_one_finished = torch.sum(self.finished_flags) > 0
    lowest_score_of_fininshed_in_finished += (
        (1. - at_least_one_finished.type(torch.float)) * -1 * Constants.INF)
    
    # non-zero value (must be less than 0) if at least one hypothesis is finished,
    # -inf if all hypothesis are not finished
    self._done = torch.lt(lower_bound_alive_scores, lowest_score_of_fininshed_in_finished)
    return self.done
  
  def _compute_topk_scores_and_seq(self, sequences, scores, scores_to_gather, flags):
    # sequences: (size * 2, ? + 1)
    # scores: (size * 2,)
    # scores_to_gather(size * 2,)
    # flags(size * 2,)
    _, topk_ids = scores.topk(self.size, 0, True, True)
    # topk_ids: (size,)
    topk_seq = torch.index_select(sequences, 0, topk_ids)
    top_log_prob = torch.index_select(scores_to_gather, 0, topk_ids)
    top_flags = torch.index_select(flags, 0, topk_ids)
    return topk_seq, top_log_prob, top_flags, topk_ids
  
  
  def grow_alive(self, curr_seq, curr_scores, curr_log_prob, curr_finished):
    # curr_seq: (size * 2, ? + 1)
    # curr_sores: (size * 2, ? + 1)
    # curr_log_prob: (size * 2, ? + 1)
    # finished_flag: (size * 2,) 1 for finished and 0 for not finished
    masked_curr_scores = curr_scores + curr_finished.type(torch.float) * self.minimal_score
    # curr_sores: (size * 2,) -inf for finished hypothesis, 0 for not finished
    
    return self._compute_topk_scores_and_seq(curr_seq, masked_curr_scores, curr_log_prob, curr_finished)
  
  def grow_finished(self, curr_seq, curr_scores, curr_finished):
    # curr_seq: (size * 2, ? + 1)
    # curr_sores: (size * 2, ? + 1)
    # finished_flag: (size * 2,) 1 for finished and 0 for not finished
    masked_curr_scores = curr_scores + (1. - curr_finished.type(torch.float)) * self.minimal_score
    # curr_sores: (size * 2,) 0 for finished hypothesis, -inf for not finished 
    
    finished_seq = torch.cat((self.finished_seq, torch.zeros((self.size, 1), dtype=torch.long, device=self.device) + self.eos_id), dim=1)
    # finished_seq: (size, ? + 1)
    curr_finished_seq = torch.cat((finished_seq, curr_seq), dim=0)
    # curr_finished_seq: (size * 2, ? + 1)
    curr_finished_scores = torch.cat((self.finished_scores, masked_curr_scores), dim=0)
    # curr_finished_scores: (size * 2, ? + 1)
    curr_finished_flags = torch.cat((self.finished_flags, curr_finished), dim=0)
    # curr_finished_flags: (size * 2, ? + 1)
    
    if (curr_finished_seq.size(1) < self.minimal_length):
      return finished_seq, self.finished_scores, self.finished_flags, None
    else:
      return self._compute_topk_scores_and_seq(curr_finished_seq, curr_finished_scores, curr_finished_scores, curr_finished_flags)
  
  def advance(self, word_prob):
    "Update beam status and check if finished or not."
    # word_prob: (size, vocab_size)
    if self.alive_seq.size()[1] == 1:
      # predict the first word
      log_probs = word_prob[0]
    else:
      log_probs = word_prob + self.alive_log_prob.view(-1, 1)
      # log_probs: (size, vocab_size)
    
    num_words = word_prob.size(1)
    
    length_penalty = math.pow((5. + self.alive_seq.size(1) / 6.), self.alpha)
    curr_scores = log_probs / length_penalty
    # curr_scores: (size, vocab_size)
    flat_curr_scores = curr_scores.view(-1)
    
    topk_scores, topk_ids = flat_curr_scores.topk(self.size * 2, 0, True, True)
    # topk_scores: (size * 2,)
    # topk_ids: (size * 2, )
    
    topk_log_probs = topk_scores * length_penalty
    # topk_log_probs: (size * 2,)
    
    topk_beam_index = topk_ids // num_words
    topk_ids %= num_words 
    # topk_beam_index: (size * 2,)
    # topk_ids: (size * 2,)
    
    topk_seq = torch.index_select(self.alive_seq, 0, topk_beam_index)
    # topk_seq: (size * 2, ?)
    topk_seq = torch.cat((topk_seq, topk_ids.view(-1, 1)), dim=1)
    # topk_seq: (size * 2, ? + 1)
    
    topk_finished = topk_ids.eq(self.eos_id)
    # topk_finished: (size * 2,)
    
    self.alive_seq, self.alive_log_prob, _, top_topk_beam_index = self.grow_alive(topk_seq, topk_scores, topk_log_probs, topk_finished)
    self.finished_seq, self.finished_scores, self.finished_flags, _ = self.grow_finished(topk_seq, topk_scores, topk_finished)
    
    self.prev_ks = torch.index_select(topk_beam_index, 0, top_topk_beam_index)
    # self.prev_ks: (size,)
    
    return self.is_finished()
    
  def get_current_state(self):
    "Get the outputs for the current timestep."
    return self.alive_seq
  
  def get_current_origin(self):
    "Get the backpointers for the current timestep."
    return self.prev_ks
  
  def get_last_target_word(self):
    return self.alive_seq[:, -1]

  @property
  def done(self):
    return self._done
  
  def get_best_hypothesis(self):
    if torch.sum(self.finished_flags) > 0:
      return self.finished_seq[0, 1:].data.cpu().numpy(), self.finished_scores[0].item()
    else:
      return self.alive_seq[0, 1:].data.cpu().numpy(), self.alive_log_prob[0].item()
