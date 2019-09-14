""" Implementation of all available options """
from __future__ import print_function

import configargparse
#from onmt.models.sru import CheckSRU


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--src_word_vec_size', '-src_word_vec_size',
              type=int, default=512,
              help='Word embedding size for src.')
    group.add('--tgt_word_vec_size', '-tgt_word_vec_size',
              type=int, default=512,
              help='Word embedding size for tgt.')

    group.add('--share_decoder_embeddings', '-share_decoder_embeddings',
              action='store_true',
              help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add('--share_embeddings', '-share_embeddings', action='store_true',
              help="""Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary for this
                       option.""")
    group.add('--position_encoding', '-position_encoding', action='store_true',
              help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add('--enc_layers', '-enc_layers', type=int, default=6,
              help='Number of layers in the encoder')
    group.add('--dec_layers', '-dec_layers', type=int, default=6,
              help='Number of layers in the decoder')
    group.add('--enc_rnn_size', '-enc_rnn_size', type=int, default=512,
              help="""Size of encoder rnn hidden states.
                       Must be equal to dec_rnn_size except for
                       speech-to-text.""")
    group.add('--dec_rnn_size', '-dec_rnn_size', type=int, default=512,
              help="""Size of decoder rnn hidden states.
                       Must be equal to enc_rnn_size except for
                       speech-to-text.""")

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add('--self_attn_type', '-self_attn_type',
              type=str, default="scaled-dot",
              help="""Self attention type in Transformer decoder
                       layer -- currently "scaled-dot" or "average" """)
    group.add('--heads', '-heads', type=int, default=8,
              help='Number of heads for transformer self-attention')
    group.add('--transformer_ff', '-transformer_ff', type=int, default=2048,
              help='Size of hidden transformer feed-forward')

def preprocess_opts(parser):
  """ Pre-procesing options """
  # Data options
  group = parser.add_argument_group('Data')

  group.add('--train_src', '-train_src', required=True,
            help="Path to the training source data")
  group.add('--train_tgt', '-train_tgt', required=True,
            help="Path to the training target data")
  group.add('--train_structure1', '-train_structure1', required=True,
            help="Path to the training structure data")
  group.add('--train_structure2', '-train_structure2', required=True,
            help="Path to the training structure data")
  group.add('--train_structure3', '-train_structure3', required=True,
            help="Path to the training structure data")
  group.add('--train_structure4', '-train_structure4', required=True,
            help="Path to the training structure data")
  group.add('--train_structure5', '-train_structure5', required=True,
            help="Path to the training structure data")
  group.add('--train_structure6', '-train_structure6', required=True,
            help="Path to the training structure data")
  group.add('--train_structure7', '-train_structure7', required=True,
            help="Path to the training structure data")
  group.add('--train_structure8', '-train_structure8', required=True,
            help="Path to the training structure data")


  group.add('--valid_src', '-valid_src', required=True,
            help="Path to the validation source data")
  group.add('--valid_tgt', '-valid_tgt', required=True,
            help="Path to the validation target data")
  group.add('--valid_structure1', '-valid_structure1', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure2', '-valid_structure2', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure3', '-valid_structure3', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure4', '-valid_structure4', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure5', '-valid_structure5', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure6', '-valid_structure6', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure7', '-valid_structure7', required=True,
            help="Path to the validation structure data")
  group.add('--valid_structure8', '-valid_structure8', required=True,
            help="Path to the validation structure data")



  group.add('--src_dir', '-src_dir', default="",
            help="Source directory for image or audio files.")

  group.add('--save_data', '-save_data', required=True,
            help="Output file for the prepared data")

  group.add('--shard_size', '-shard_size', type=int, default=1000000,
            help="""Divide src_corpus and tgt_corpus into
                     smaller multiple src_corpus and tgt corpus files, then
                     build shards, each shard will have
                     opt.shard_size samples except last shard.
                     shard_size=0 means no segmentation
                     shard_size>0 means segment dataset into multiple shards,
                     each shard has shard_size samples""")

  # Dictionary options, for text corpus

  group = parser.add_argument_group('Vocab')
  group.add('--src_vocab', '-src_vocab', default="",
            help="""Path to an existing source vocabulary. Format:
                     one word per line.""")
  group.add('--tgt_vocab', '-tgt_vocab', default="",
            help="""Path to an existing target vocabulary. Format:
                     one word per line.""")
  group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
            type=str, default='',
            help="Path prefix to existing features vocabularies")
  group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
            help="Size of the source vocabulary")
  group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
            help="Size of the target vocabulary")
  group.add('--structure_vocab_size', '-structure_vocab_size', type=int, default=50000,
            help="Size of the structure_vocab_size")

  group.add('--src_words_min_frequency',
            '-src_words_min_frequency', type=int, default=0)
  group.add('--tgt_words_min_frequency',
            '-tgt_words_min_frequency', type=int, default=0)
  group.add('--structure_words_min_frequency',
            '-structure_words_min_frequency', type=int, default=0)
  group.add('--share_vocab', '-share_vocab', action='store_true',
            help="Share source and target vocabulary")

  # Truncation options, for text corpus
  group = parser.add_argument_group('Pruning')
  group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
            help="Maximum source sequence length")
  group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
            type=int, default=0,
            help="Truncate source sequence length.")
  group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
            help="Maximum target sequence length to keep.")
  group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
            type=int, default=0,
            help="Truncate target sequence length.")
  group.add('--lower', '-lower', action='store_true', help='lowercase data')

  # Data processing options
  group = parser.add_argument_group('Random')
  group.add('--shuffle', '-shuffle', type=int, default=0,
            help="Shuffle data")
  group.add('--seed', '-seed', type=int, default=3435,
            help="Random seed")

  group = parser.add_argument_group('Logging')
  group.add('--report_every', '-report_every', type=int, default=100000,
            help="Report status every this many sentences")
  group.add('--log_file', '-log_file', type=str, default="",
            help="Output logs to a file under this path.")
  group.add('--log_file_level', '-log_file_level', type=str, default="0")

def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add('--data', '-data', required=True,
              help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")

    group.add('--save_model', '-save_model', default='model',
              help="""Model filename (the model will be saved as
                       <save_model>_N.pt where N is the number
                       of steps""")

    group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
              type=int, default=5000,
              help="""Save a checkpoint every X steps""")
    group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
              help="""Keep X checkpoints (negative: keep all)""")

    # GPU
    group.add('--gpuid', '-gpuid', default=[], nargs='*', type=int,
              help="Deprecated see world_size and gpu_ranks.")    #nargs='*' 表示参数可设置零个或多个
                                                                  #nargs='+' 表示参数可设置一个或多个
    group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
              help="list of ranks of each process.")
    group.add('--world_size', '-world_size', default=1, type=int,
              help="total number of distributed processes.")
    group.add('--gpu_backend', '-gpu_backend',
              default="nccl", type=str,
              help="Type of torch distributed backend")
    group.add('--gpu_verbose_level', '-gpu_verbose_level', default=0, type=int,
              help="Gives more info on each process per GPU.")
    group.add('--master_ip', '-master_ip', default="localhost", type=str,
              help="IP of master for torch.distributed training.")
    group.add('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")

    group.add('--seed', '-seed', type=int, default=-1,
              help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add('--param_init', '-param_init', type=float, default=0.1,
              help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add('--param_init_glorot', '-param_init_glorot', action='store_true',
              help="""Init parameters with xavier_uniform.
                       Required for transfomer.""")

    group.add('--train_from', '-train_from', default='', type=str,
              help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
    group.add('--reset_optim', '-reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="""Optimization resetter when train_from.""")

    # Pretrained word vectors
    group.add('--pre_word_vecs_enc', '-pre_word_vecs_enc',
              help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add('--pre_word_vecs_dec', '-pre_word_vecs_dec',
              help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add('--fix_word_vecs_enc', '-fix_word_vecs_enc',
              action='store_true',
              help="Fix word embeddings on the encoder side.")
    group.add('--fix_word_vecs_dec', '-fix_word_vecs_dec',
              action='store_true',
              help="Fix word embeddings on the decoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
    group.add('--normalization', '-normalization', default='sents',
              choices=["sents", "tokens"],
              help='Normalization method of the gradient.')
    group.add('--accum_count', '-accum_count', type=int, default=1,
              help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")
    group.add('--valid_steps', '-valid_steps', type=int, default=10000,
              help='Perfom validation every X steps')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
              help='Maximum batch size for validation')
    group.add('--max_generator_batches', '-max_generator_batches',
              type=int, default=32,
              help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    group.add('--train_steps', '-train_steps', type=int, default=100000,
              help='Number of training steps')
    group.add('--epochs', '-epochs', type=int, default=0,
              help='Deprecated epochs see train_steps')
    group.add('--optim', '-optim', default='sgd',
              choices=['sgd', 'adagrad', 'adadelta', 'adam',
                       'sparseadam'],
              help="""Optimization method.""")
    group.add('--adagrad_accumulator_init', '-adagrad_accumulator_init',
              type=float, default=0,
              help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
    group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
              help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add('--dropout', '-dropout', type=float, default=0.3,
              help="Dropout probability; applied in LSTM stacks.")
    group.add('--truncated_decoder', '-truncated_decoder', type=int, default=0,
              help="""Truncated bptt.""")
    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
              help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
              help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=0.5,
              help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) steps have gone past
                       start_decay_steps""")
    group.add('--start_decay_steps', '-start_decay_steps',
              type=int, default=50000,
              help="""Start decaying every decay_steps after
                       start_decay_steps""")
    group.add('--decay_steps', '-decay_steps', type=int, default=10000,
              help="""Decay every decay_steps""")

    group.add('--decay_method', '-decay_method', type=str, default="none",
              choices=['noam', 'none'], help="Use a custom decay rate.")
    group.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
              help="""Number of warmup steps for custom decay.""")


    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=50,
              help="Print stats at this interval.")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              default="0")
    group.add('--exp_host', '-exp_host', type=str, default="",
              help="Send logs to this crayon server.")
    group.add('--exp', '-exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add('--tensorboard', '-tensorboard', action="store_true",
              help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("-tensorboard_log_dir", type=str,
                       default="runs/onmt",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)

def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help='Path to model .pt file(s). '
              'Multiple models can be specified, '
              'for ensemble decoding.')

    group = parser.add_argument_group('Data')

    group.add('--src', '-src', required=True, help="""Source sequence to decode (one line per
                       sequence)""")
    group.add('--tgt', '-tgt', help='True target sequence (optional)')

    group.add('--structure1', '-structure1', help='structure1')
    group.add('--structure2', '-structure2', help='structure2')
    group.add('--structure3', '-structure3', help='structure3')
    group.add('--structure4', '-structure4', help='structure4')
    group.add('--structure5', '-structure5', help='structure5')
    group.add('--structure6', '-structure6', help='structure6')
    group.add('--structure7', '-structure7', help='structure7')
    group.add('--structure8', '-structure8', help='structure8')



    group.add('--output', '-output', default='pred.txt',
              help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add('--beam_size', '-beam_size', type=int, default=5,
              help='Beam size')
    group.add('--min_length', '-min_length', type=int, default=0,
              help='Minimum prediction length')
    group.add('--max_length', '-max_length', type=int, default=100,
              help='Maximum prediction length.')
    group.add('--decode_extra_length', '-decode_extra_length', type=int, default=50,
              help='Maximum extra prediction length.')
    group.add('--decode_min_length', '-decode_min_length', type=int, default=-1,
              help="""Minimum extra prediction length. 
                    -1: no miximum limitation on the prediction length;
                    0: minimum length is the source length
                    otherwise minimum length is source length - this many""")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add('--length_penalty', '-length_penalty', default='none',
              choices=['none', 'wu', 'avg'],
              help="""Length Penalty to use.""")
    group.add('--alpha', '-alpha', type=float, default=0.,
              help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add('--beta', '-beta', type=float, default=-0.,
              help="""Coverage penalty parameter""")

    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              default="0")
    group.add('--dump_beam', '-dump_beam', type=str, default="",
              help='File to dump beam information to.')

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=30,
              help='Batch size')
    group.add('--gpu', '-gpu', type=int, default=-1,
                       help="Device to run on")