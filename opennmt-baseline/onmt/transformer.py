"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.constants as Constants 

from onmt.transformer_encoder import TransformerEncoder
from onmt.transformer_decoder import TransformerDecoder

from onmt.embeddings import Embeddings
from utils.misc import use_gpu
from utils.logging import logger
from inputters.dataset import load_fields_from_vocab

class NMTModel(nn.Module):
  def __init__(self, encoder, decoder):
    super(NMTModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src, tgt, lengths):
    tgt = tgt[:-1]  # exclude last target from inputs

    _, memory_bank, lengths = self.encoder(src, lengths)
    self.decoder.init_state(src, memory_bank)
    dec_out, attns = self.decoder(tgt)

    return dec_out, attns

def build_embeddings(opt, word_dict, for_encoder=True):
  """
  Build an Embeddings instance.
  Args:
      opt: the option in current environment.
      word_dict(Vocab): words dictionary.
      feature_dicts([Vocab], optional): a list of feature dictionary.
      for_encoder(bool): build Embeddings for encoder or decoder?
  """
  if for_encoder:
    embedding_dim = opt.src_word_vec_size
  else:
    embedding_dim = opt.tgt_word_vec_size

  word_padding_idx = word_dict.stoi[Constants.PAD_WORD]
  num_word_embeddings = len(word_dict)

  return Embeddings(word_vec_size=embedding_dim,
                    position_encoding=opt.position_encoding,
                    dropout=opt.dropout,
                    word_padding_idx=word_padding_idx,
                    word_vocab_size=num_word_embeddings,
                    sparse=opt.optim == "sparseadam")


def build_encoder(opt, embeddings):
  """
  Various encoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this encoder.
  """
  return TransformerEncoder(opt.enc_layers, opt.enc_rnn_size,
                            opt.heads, opt.transformer_ff,
                            opt.dropout, embeddings)

def build_decoder(opt, embeddings):
  """
  Various decoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this decoder.
  """
  return TransformerDecoder(opt.dec_layers, opt.dec_rnn_size,
                     opt.heads, opt.transformer_ff,
                     opt.dropout, embeddings)

def load_test_model(opt, dummy_opt, model_path=None):
  if model_path is None:
    model_path = opt.models[0]
  checkpoint = torch.load(model_path,
                        map_location=lambda storage, loc: storage)
  fields = load_fields_from_vocab(checkpoint['vocab'])

  model_opt = checkpoint['opt']

  for arg in dummy_opt:
    if arg not in model_opt:
      model_opt.__dict__[arg] = dummy_opt[arg]
  model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
  model.eval()
  model.generator.eval()
  return fields, model


def build_base_model(model_opt, fields, gpu, checkpoint=None):
  """
  Args:
      model_opt: the option loaded from checkpoint.
      fields: `Field` objects for the model.
      gpu(bool): whether to use gpu.
      checkpoint: the model gnerated by train phase, or a resumed snapshot
                  model from a stopped training.
  Returns:
      the NMTModel.
  """

  # for backward compatibility
  if model_opt.enc_rnn_size != model_opt.dec_rnn_size:
    raise AssertionError("""We do not support different encoder and
                         decoder rnn sizes for translation now.""")

  # Build encoder.
  src_dict = fields["src"].vocab
  src_embeddings = build_embeddings(model_opt, src_dict)
  encoder = build_encoder(model_opt, src_embeddings)

  # Build decoder.
  tgt_dict = fields["tgt"].vocab
  tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                    for_encoder=False)

  # Share the embedding matrix - preprocess with share_vocab required.
  if model_opt.share_embeddings:
    # src/tgt vocab should be the same if `-share_vocab` is specified.
    if src_dict != tgt_dict:
      raise AssertionError('The `-share_vocab` should be set during '
                           'preprocess if you use share_embeddings!')

    tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

  decoder = build_decoder(model_opt, tgt_embeddings)

  # Build NMTModel(= encoder + decoder).
  device = torch.device("cuda:0" if gpu else "cpu")
  model = NMTModel(encoder, decoder)

  # Build Generator.
  gen_func = nn.LogSoftmax(dim=-1)
  generator = nn.Sequential(
    nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab)),
    gen_func
  )
  if model_opt.share_decoder_embeddings:
    generator[0].weight = decoder.embeddings.word_lut.weight

  # Load the model states from checkpoint or initialize them.
  if checkpoint is not None:
    # This preserves backward-compat for models using customed layernorm
    def fix_key(s):
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                 r'\1.layer_norm\2.bias', s)
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                 r'\1.layer_norm\2.weight', s)
      return s

    checkpoint['model'] = \
      {fix_key(k): v for (k, v) in checkpoint['model'].items()}
    # end of patch for backward compatibility

    model.load_state_dict(checkpoint['model'], strict=False)
    generator.load_state_dict(checkpoint['generator'], strict=False)
  else:
    if model_opt.param_init != 0.0:
      for p in model.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
      for p in generator.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
      for p in model.parameters():
        if p.dim() > 1:
          xavier_uniform_(p)
      for p in generator.parameters():
        if p.dim() > 1:
          xavier_uniform_(p)

    if hasattr(model.encoder, 'embeddings'):
      model.encoder.embeddings.load_pretrained_vectors(
          model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
    if hasattr(model.decoder, 'embeddings'):
      model.decoder.embeddings.load_pretrained_vectors(
          model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
  #pdb.set_trace()
  # Add generator to model (this registers it as parameter of model).
  model.generator = generator
  model.to(device)

  return model


def build_model(model_opt, opt, fields, checkpoint):
  """ Build the Model """
  logger.info('Building model...')
  model = build_base_model(model_opt, fields,
                           use_gpu(opt), checkpoint)
  logger.info(model)
  return model
