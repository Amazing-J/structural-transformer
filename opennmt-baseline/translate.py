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
  translator.translate(src_data_iter=src_iter,
                       tgt_data_iter=tgt_iter,
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
