"""Run evaluation."""

import os
import pathlib

import flax
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from PIL import Image
from tqdm import tqdm

from model import Model
from utils import calculate_psnr, ensure_path, is_image_file, save_img

# disables the preallocation behavior
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', 'Zero-DCE', ['Zero-DCE', 'Zero-DCE++'],
                  'Model variant')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.', required=True)
flags.DEFINE_string('input_dir', './test_data', 'Input dir to the test set')
flags.DEFINE_string('output_dir', 'outputs',
                    'Output dir to store predicted images')
flags.DEFINE_string('target_dir', '', 'Path to gt images')
flags.DEFINE_boolean('has_target', False, 'Whether has corresponding gt image.')
flags.DEFINE_boolean('save_images', True, 'Dump predicted images.')


def main(_):
  data_root = pathlib.Path(FLAGS.input_dir)
  input_img_paths = [p for p in data_root.glob('**/*') if is_image_file(p)]

  if FLAGS.has_target:
    target_img_paths = [
        pathlib.Path(FLAGS.target_dir).joinpath(*p.parts[1:])
        for p in input_img_paths
    ]

  model = Model(FLAGS.model)
  ckpt = checkpoints.restore_checkpoint(FLAGS.ckpt_path, target=None)
  params = ckpt['params']

  psnr_all = []

  image_loader = lambda p: np.asarray(Image.open(p).convert('RGB')) / 255.

  for i, fpath in enumerate(tqdm(input_img_paths, desc='Testing')):
    input_img = image_loader(fpath)
    input_img = np.expand_dims(input_img, axis=0)

    if FLAGS.has_target:
      target_file = target_img_paths[i]
      target_img = image_loader(target_file)

    preds, _ = model.apply({'params': flax.core.freeze(params)}, input_img)
    preds = np.array(preds[0], np.float32)

    if FLAGS.has_target:
      psnr = calculate_psnr(target_img * 255.,
                            preds * 255.,
                            crop_border=0,
                            test_y_channel=False)
      psnr_all.append(psnr)

    # save files
    if FLAGS.save_images:
      save_path = pathlib.Path(FLAGS.output_dir).joinpath(*fpath.parts[1:])
      ensure_path(save_path)
      save_img(preds, save_path)

  if psnr_all:
    print(f'average psnr = {sum(psnr_all)/len(psnr_all):.4f}')


if __name__ == '__main__':
  app.run(main)
