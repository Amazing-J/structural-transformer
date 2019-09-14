from itertools import chain
import gc
import glob
import codecs
import math
from collections import defaultdict

import torch
import torchtext.data
from utils.logging import logger
import onmt.constants as Constants

def _getstate(self):
  return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
  self.__dict__.update(state)
  self.stoi = defaultdict(lambda: 0, self.stoi)

torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def make_text_iterator_from_file(path):
  with codecs.open(path, "r", "utf-8") as corpus_file:
    for line in corpus_file:
      yield line     #每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用

def make_features(batch, side):
  """
  Args:
      batch (Tensor): a batch of source or target or structure data.
      side (str): for source or for target or for structure.
  Returns:
      A sequence of src/tgt tensors with optional feature tensors of size (len x batch).
  """
  assert side in ['src', 'tgt', 'structure1', 'structure2', 'structure3', 'structure4', 'structure5', 'structure6','structure7','structure8']
  if isinstance(batch.__dict__[side], tuple):   #isinstance()来判断一个对象是否是一个已知的类型
    data = batch.__dict__[side][0]
  else:
    data = batch.__dict__[side]

  return data

def save_fields_to_vocab(fields):
  """
  Save Vocab objects in Field objects to `vocab.pt` file.
  """
  vocab = []
  for k, f in fields.items():
    if f is not None and 'vocab' in f.__dict__:
      f.vocab.stoi = f.vocab.stoi
      vocab.append((k, f.vocab))
  return vocab

def get_source_fields(fields=None):
  if fields is None:
    fields = {}

  fields["src"] = torchtext.data.Field(
    pad_token=Constants.PAD_WORD,
    eos_token=Constants.EOS_WORD,
    include_lengths=True)    #include_lengths字段为True可以在返回minibatch的时候同时返回一个表示每个句子长度的list

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_target_fields(fields=None):
  if fields is None:
    fields = {}

  fields["tgt"] = torchtext.data.Field(
    init_token=Constants.BOS_WORD, 
    eos_token=Constants.EOS_WORD,
    pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_structure1_fields(fields=None):
  if fields is None:
    fields = {}

  nesting_field = torchtext.data.Field(
    pad_token=Constants.PAD_WORD)

  fields["structure1"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_structure2_fields(fields=None):
  if fields is None:
    fields = {}

  nesting_field = torchtext.data.Field(
    pad_token=Constants.PAD_WORD)

  fields["structure2"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_structure3_fields(fields=None):
  if fields is None:
    fields = {}

  nesting_field = torchtext.data.Field(
    pad_token=Constants.PAD_WORD)

  fields["structure3"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_structure4_fields(fields=None):
  if fields is None:
    fields = {}

  nesting_field = torchtext.data.Field(
    pad_token=Constants.PAD_WORD)

  fields["structure4"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_structure5_fields(fields=None):
  if fields is None:
    fields = {}

  nesting_field = torchtext.data.Field(
    pad_token=Constants.PAD_WORD)

  fields["structure5"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

# def get_structure6_fields(fields=None):
#   if fields is None:
#     fields = {}
#
#   nesting_field = torchtext.data.Field(
#     pad_token=Constants.PAD_WORD)
#
#   fields["structure6"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)
#
#   fields["indices"] = torchtext.data.Field(
#       use_vocab=False, dtype=torch.long,
#       sequential=False)
#
#   return fields
#
# def get_structure7_fields(fields=None):
#   if fields is None:
#     fields = {}
#
#   nesting_field = torchtext.data.Field(
#     pad_token=Constants.PAD_WORD)
#
#   fields["structure7"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)
#
#   fields["indices"] = torchtext.data.Field(
#       use_vocab=False, dtype=torch.long,
#       sequential=False)
#
#   return fields
#
# def get_structure8_fields(fields=None):
#   if fields is None:
#     fields = {}
#
#   nesting_field = torchtext.data.Field(
#     pad_token=Constants.PAD_WORD)
#
#   fields["structure8"] = torchtext.data.NestedField(nesting_field, pad_token=Constants.PAD_WORD)
#
#   fields["indices"] = torchtext.data.Field(
#       use_vocab=False, dtype=torch.long,
#       sequential=False)
#
#   return fields


def get_fields():
  fields = {}
    
  fields = get_source_fields(fields)
  fields = get_target_fields(fields)
  fields = get_structure1_fields(fields)
  fields = get_structure2_fields(fields)
  fields = get_structure3_fields(fields)
  fields = get_structure4_fields(fields)
  fields = get_structure5_fields(fields)
  # fields = get_structure6_fields(fields)
  # fields = get_structure7_fields(fields)
  # fields = get_structure8_fields(fields)

  return fields

def load_fields_from_vocab(vocab):
  """
  Load Field objects from `vocab.pt` file.
  """
  vocab = dict(vocab)
  fields = get_fields()
  for k, v in vocab.items():
    # Hack. Can't pickle defaultdict :(
    v.stoi = defaultdict(lambda: 0, v.stoi)
    fields[k].vocab = v
  return fields

def load_fields(opt, checkpoint):
  if checkpoint is not None:
    logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
    fields = load_fields_from_vocab(checkpoint['vocab'])
  else:
    fields = load_fields_from_vocab(torch.load(opt.data + '_vocab.pt'))
  fields['structure1'].nesting_field.vocab = fields['structure1'].vocab
  fields['structure2'].nesting_field.vocab = fields['structure2'].vocab
  fields['structure3'].nesting_field.vocab = fields['structure3'].vocab
  fields['structure4'].nesting_field.vocab = fields['structure4'].vocab
  fields['structure5'].nesting_field.vocab = fields['structure5'].vocab
  # fields['structure6'].nesting_field.vocab = fields['structure6'].vocab
  # fields['structure7'].nesting_field.vocab = fields['structure7'].vocab
  # fields['structure8'].nesting_field.vocab = fields['structure8'].vocab
  logger.info(' * vocabulary size. source = %d; target = %d; structure1 = %d; structure2 = %d; structure3 = %d; structure4 = %d; structure5 = %d; '
              %
              (len(fields['src'].vocab),
               len(fields['tgt'].vocab),
               len(fields['structure1'].nesting_field.vocab),
               len(fields['structure2'].nesting_field.vocab),
               len(fields['structure3'].nesting_field.vocab),
               len(fields['structure4'].nesting_field.vocab),
               len(fields['structure5'].nesting_field.vocab)))

  return fields

class DatasetIter(object):
  """ An Ordered Dataset Iterator, supporting multiple datasets,
      and lazy loading.

  Args:
      datsets (list): a list of datasets, which are lazily loaded.
      fields (dict): fields dict for the datasets.
      batch_size (int): batch size.
      batch_size_fn: custom batch process function.
      device: the GPU device.
      is_train (bool): train or valid?
  """

  def __init__(self, datasets, fields, batch_size, batch_size_fn,
               device, is_train):
    self.datasets = datasets
    self.fields = fields
    self.batch_size = batch_size
    self.batch_size_fn = batch_size_fn
    self.device = device
    self.is_train = is_train

    self.cur_iter = self._next_dataset_iterator(datasets)
    # We have at least one dataset.
    assert self.cur_iter is not None

  def __iter__(self):
    dataset_iter = (d for d in self.datasets)

    while self.cur_iter is not None:
      for batch in self.cur_iter:
        yield batch
      self.cur_iter = self._next_dataset_iterator(dataset_iter)

  def __len__(self):
    # We return the len of cur_dataset, otherwise we need to load
    # all datasets to determine the real len, which loses the benefit
    # of lazy loading.
    assert self.cur_iter is not None
    return len(self.cur_iter)

  def _next_dataset_iterator(self, dataset_iter):
    try:
      # Drop the current dataset for decreasing memory
      if hasattr(self, "cur_dataset"):
        self.cur_dataset.examples = None
        gc.collect()
        del self.cur_dataset
        gc.collect()

      self.cur_dataset = next(dataset_iter)
    except StopIteration:
      return None

    # We clear `fields` when saving, restore when loading.
    self.cur_dataset.fields = self.fields

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    return OrderedIterator(
      dataset=self.cur_dataset, batch_size=self.batch_size,
      batch_size_fn=self.batch_size_fn,
      device=self.device, train=self.is_train,
      sort=False, sort_within_batch=True,
      repeat=False)
    
class OrderedIterator(torchtext.data.Iterator):
  """ Ordered Iterator Class """

  def create_batches(self):
    """ Create batches """
    if self.train:
      def _pool(data, random_shuffler):
        for p in torchtext.data.batch(data, self.batch_size * 100):
          p_batch = torchtext.data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
          for b in random_shuffler(list(p_batch)):
            yield b

      self.batches = _pool(self.data(), self.random_shuffler)
    else:
      self.batches = []
      for b in torchtext.data.batch(self.data(), self.batch_size, self.batch_size_fn):
        self.batches.append(sorted(b, key=self.sort_key))





def load_dataset(corpus_type, opt):
  assert corpus_type in ["train", "valid"]

  def _dataset_loader(pt_file, corpus_type):
    dataset = torch.load(pt_file)
    logger.info('Loading %s dataset from %s, number of examples: %d' %
                (corpus_type, pt_file, len(dataset)))
    return dataset

  # Sort the glob output by file name (by increasing indexes).
  pts = sorted(glob.glob(opt.data + '_' + corpus_type + '.[0-9]*.pt'))
  if pts:
    for pt in pts:
      yield _dataset_loader(pt, corpus_type)
  else:
    pt = opt.data + '_' + corpus_type + '.pt'
    yield _dataset_loader(pt, corpus_type)




def build_dataset(fields,
                  src_data_iter,
                  tgt_data_iter,
                  structure_data_iter1,
                  structure_data_iter2,
                  structure_data_iter3,
                  structure_data_iter4,
                  structure_data_iter5,
                  src_seq_length=0,
                  tgt_seq_length=0,
                  src_seq_length_trunc=0,
                  tgt_seq_length_trunc=0,
                  use_filter_pred=True):
  assert src_data_iter != None
  src_examples_iter = Dataset.make_examples(src_data_iter, src_seq_length_trunc, "src")
  
  if tgt_data_iter != None:
    tgt_examples_iter = Dataset.make_examples(tgt_data_iter, tgt_seq_length_trunc, "tgt")
  else:
    tgt_examples_iter = None

  if structure_data_iter1 != None:
    structure_examples_iter1 = Dataset.make_nested_examples(structure_data_iter1, None, "structure1")
  else:
    structure_examples_iter1 = None

  if structure_data_iter2 != None:
    structure_examples_iter2 = Dataset.make_nested_examples(structure_data_iter2, None, "structure2")
  else:
    structure_examples_iter2 = None

  if structure_data_iter3 != None:
    structure_examples_iter3 = Dataset.make_nested_examples(structure_data_iter3, None, "structure3")
  else:
    structure_examples_iter3 = None

  if structure_data_iter4 != None:
    structure_examples_iter4 = Dataset.make_nested_examples(structure_data_iter4, None, "structure4")
  else:
    structure_examples_iter4 = None

  if structure_data_iter5 != None:
    structure_examples_iter5 = Dataset.make_nested_examples(structure_data_iter5, None, "structure5")
  else:
    structure_examples_iter5 = None

  # if structure_data_iter6 != None:
  #   structure_examples_iter6 = Dataset.make_nested_examples(structure_data_iter6, None, "structure6")
  # else:
  #   structure_examples_iter6 = None
  #
  # if structure_data_iter7 != None:
  #   structure_examples_iter7 = Dataset.make_nested_examples(structure_data_iter7, None, "structure7")
  # else:
  #   structure_examples_iter7 = None
  #
  # if structure_data_iter8 != None:
  #   structure_examples_iter8 = Dataset.make_nested_examples(structure_data_iter8, None, "structure8")
  # else:
  #   structure_examples_iter8 = None



  dataset = Dataset(fields, src_examples_iter, tgt_examples_iter,
                    structure_examples_iter1,
                    structure_examples_iter2,
                    structure_examples_iter3,
                    structure_examples_iter4,
                    structure_examples_iter5,
                    src_seq_length=src_seq_length,
                    tgt_seq_length=tgt_seq_length,
                    use_filter_pred=use_filter_pred)

  return dataset






def build_dataset_iter(datasets, fields, opt, is_train=True):
  """
  This returns user-defined train/validate data iterator for the trainer
  to iterate over. We implement simple ordered iterator strategy here,
  but more sophisticated strategy like curriculum learning is ok too.
  """
  batch_size = opt.batch_size if is_train else opt.valid_batch_size
  if is_train and opt.batch_type == "tokens":
    def batch_size_fn(new, count, sofar):
      """
      In token batching scheme, the number of sequences is limited
      such that the total number of src/tgt tokens (including padding)
      in a batch <= batch_size
      """
      # Maintains the longest src and tgt length in the current batch
      global max_src_in_batch, max_tgt_in_batch
      # Reset current longest length at a new batch (count=1)
      if count == 1:
          max_src_in_batch = 0
          max_tgt_in_batch = 0
      # Src: <bos> w1 ... wN <eos>
      max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
      # Tgt: w1 ... wN <eos>
      max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
      src_elements = count * max_src_in_batch
      tgt_elements = count * max_tgt_in_batch
      return max(src_elements, tgt_elements)
  else:
    batch_size_fn = None

  if opt.gpu_ranks:
    device = "cuda"
  else:
    device = "cpu"

  return DatasetIter(datasets, fields, batch_size, batch_size_fn, device, is_train)


class Dataset(torchtext.data.Dataset):
  def __init__(self, fields,
               src_examples_iter,
               tgt_examples_iter,
               structure_examples_iter1,
               structure_examples_iter2,
               structure_examples_iter3,
               structure_examples_iter4,
               structure_examples_iter5,
               src_seq_length=0, tgt_seq_length=0,
               use_filter_pred=True):

    self.src_vocabs = []
    
    def _join_dicts(*args):
      return dict(chain(*[d.items() for d in args]))

    out_fields = get_source_fields()
    if tgt_examples_iter is not None and structure_examples_iter1 is not None :
      examples_iter = (_join_dicts(src, tgt, structure1, structure2, structure3, structure4, structure5)
                       for src, tgt, structure1, structure2, structure3, structure4, structure5
                       in zip(src_examples_iter,
                              tgt_examples_iter,
                              structure_examples_iter1,
                              structure_examples_iter2,
                              structure_examples_iter3,
                              structure_examples_iter4,
                              structure_examples_iter5,
                              ))
      out_fields = get_target_fields(out_fields)
      out_fields = get_structure1_fields(out_fields)
      out_fields = get_structure2_fields(out_fields)
      out_fields = get_structure3_fields(out_fields)
      out_fields = get_structure4_fields(out_fields)
      out_fields = get_structure5_fields(out_fields)

    else:
      # examples_iter = src_examples_iter
      examples_iter = (_join_dicts(src, structure1, structure2, structure3, structure4, structure5)
                       for src, structure1, structure2, structure3, structure4, structure5
                       in zip(src_examples_iter,
                              structure_examples_iter1,
                              structure_examples_iter2,
                              structure_examples_iter3,
                              structure_examples_iter4,
                              structure_examples_iter5,
                              ))
      out_fields = get_structure1_fields(out_fields)
      out_fields = get_structure2_fields(out_fields)
      out_fields = get_structure3_fields(out_fields)
      out_fields = get_structure4_fields(out_fields)
      out_fields = get_structure5_fields(out_fields)


      fields['structure1'].nesting_field.vocab = fields['structure1'].vocab
      fields['structure2'].nesting_field.vocab = fields['structure2'].vocab
      fields['structure3'].nesting_field.vocab = fields['structure3'].vocab
      fields['structure4'].nesting_field.vocab = fields['structure4'].vocab
      fields['structure5'].nesting_field.vocab = fields['structure5'].vocab


      
    keys = out_fields.keys()   #dict_keys(['src', 'indices', 'tgt', 'structure','index'])

    out_fields = [(k, fields[k]) for k in keys]
    example_values = ([ex[k] for k in keys] for ex in examples_iter)
    out_examples = []

    for ex_values in example_values:
      example = torchtext.data.Example()
      for (name, field), val in zip(out_fields, ex_values):

        if field is not None:
          setattr(example, name, field.preprocess(val))    #setattr(object, name, values)给对象的属性赋值，若属性不存在，先创建再赋值。
          #print(name,val)  #name: src,tgt,structure   val: tuple(value)
        else:
          setattr(example, name, val)
      out_examples.append(example)


    def filter_pred(example):
      """ 只使用filter_pred值为True的example """
      return 0 < len(example.src) <= src_seq_length and 0 < len(example.tgt) <= tgt_seq_length

    filter_pred = filter_pred if use_filter_pred else lambda x: True

    # for line in out_examples:
    #   print(line.__dict__)

    super(Dataset, self).__init__(out_examples, out_fields, filter_pred)
  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, _d):
    self.__dict__.update(_d)
    
  def sort_key(self, ex):
    if hasattr(ex, "tgt"):
      return len(ex.src), len(ex.tgt)
    return len(ex.src)

  @staticmethod
  def make_nested_examples(text_iter, truncate, side):
    for i, line in enumerate(text_iter):
      line = line.strip().split()
      length = len(line)
      src_length = int (math.sqrt(length))
      words = (line[j: j+src_length] for j in range(0, len(line), src_length))      # 每行代表一个concept和所有concept关系
      example_dict = {side: tuple(words), "indices": i}
      yield example_dict

  @staticmethod
  def make_examples(text_iter, truncate, side):   #side={src/tgt}
    for i, line in enumerate(text_iter):
      words = line.strip().split()
      if truncate:
        words = words[:truncate]

      example_dict = {side: tuple(words), "indices": i}
      yield example_dict
