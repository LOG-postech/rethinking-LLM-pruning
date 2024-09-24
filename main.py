from absl import app, flags
from ml_collections import config_flags

from llama import main as main_llama
from opt import main as main_opt

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)

def main(argv):
  if 'opt' in FLAGS.config.model:
    main_opt(FLAGS.config)
  elif 'llama' in FLAGS.config.model:
    main_llama(FLAGS.config)
  else:
    raise ValueError("Invalid model")

if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)