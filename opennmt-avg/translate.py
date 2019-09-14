#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse
import codecs

from utils.logging import init_logger
from inputters.dataset import make_text_iterator_from_file
import onmt.opts as opts
from onmt.translator import build_translator

def main(opt):
  
  translator = build_translator(opt)
  out_file = codecs.open(opt.output, 'w+', 'utf-8')
  
  src_iter = make_text_iterator_from_file(opt.src)

  if opt.tgt is not None:
    tgt_iter = make_text_iterator_from_file(opt.tgt)
  else:
    tgt_iter = None

  if opt.structure1 is not None:
    structure_iter1 = make_text_iterator_from_file(opt.structure1)
  else:
    structure_iter1 = None

  if opt.structure2 is not None:
    structure_iter2 = make_text_iterator_from_file(opt.structure2)
  else:
    structure_iter2 = None

  if opt.structure3 is not None:
    structure_iter3 = make_text_iterator_from_file(opt.structure3)
  else:
    structure_iter3 = None

  if opt.structure4 is not None:
    structure_iter4 = make_text_iterator_from_file(opt.structure4)
  else:
    structure_iter4 = None

  if opt.structure5 is not None:
    structure_iter5 = make_text_iterator_from_file(opt.structure5)
  else:
    structure_iter5 = None




  translator.translate(src_data_iter=src_iter,
                       tgt_data_iter=tgt_iter,
                       structure_iter1=structure_iter1,
                       structure_iter2=structure_iter2,
                       structure_iter3=structure_iter3,
                       structure_iter4=structure_iter4,
                       structure_iter5=structure_iter5,
                       batch_size=opt.batch_size,
                       out_file=out_file)
  out_file.close()


if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
    description='translate.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  opts.config_opts(parser)
  opts.translate_opts(parser)

  opt = parser.parse_args()
  logger = init_logger(opt.log_file)
  logger.info("Input args: %r", opt)
  main(opt)
