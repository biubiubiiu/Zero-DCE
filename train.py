"""Train a model"""

import logging
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import app, flags
from addict import Dict
from flax.training import checkpoints, train_state
from tqdm import tqdm

from data import build_dataset
from model import Model
from utils import ensure_path, setup_logger

FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'Zero-DCE', ['Zero-DCE', 'Zero-DCE++'],
                  'Model variant')
flags.DEFINE_string('train_dir', '/home/hjj/data/Zero-DCE-train',
                    'Path to training data')
flags.DEFINE_string('work_dir', 'work_dir', 'Path to save logs and models')

CONFIG = None


def parse_config():
  train_config = {
      'Zero-DCE': {
          'lr': 1e-4,
          'weight_decay': 1e-4,  # L2 penalty of weights
          'grad_clip_norm': 0.1,
          'num_epochs': 200,
          'batch_size': 8,
          'image_size': 256,
          'loss_tv_weight': 200,
          'loss_spa_weight': 1,
          'loss_col_weight': 5,
          'loss_exp_weight': 4,
      },
      'Zero-DCE++': {
          'lr': 1e-4,
          'weight_decay': 1e-4,
          'grad_clip_norm': 0.1,
          'num_epochs': 100,
          'batch_size': 8,
          'image_size': 512,
          'loss_tv_weight': 1600,
          'loss_spa_weight': 1,
          'loss_col_weight': 5,
          'loss_exp_weight': 4,
      }
  }[FLAGS.model]

  global CONFIG
  CONFIG = Dict()
  CONFIG.update(train_config)
  CONFIG.update(FLAGS.flag_values_dict())


@jax.jit
def train_step(state, batch):

  def loss_spatial(original, enhanced, rsize=4):
    kernel = jnp.array([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                         [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                       dtype=jnp.float32)

    ori_mean = jnp.mean(original, axis=-1, keepdims=True)
    enh_mean = jnp.mean(enhanced, axis=-1, keepdims=True)

    ori_pooled = nn.avg_pool(ori_mean,
                             window_shape=(rsize, rsize),
                             strides=(rsize, rsize))
    enh_pooled = nn.avg_pool(enh_mean,
                             window_shape=(rsize, rsize),
                             strides=(rsize, rsize))

    ori_grad = jax.lax.conv(
        jnp.transpose(ori_pooled, [0, 3, 1, 2]),  # to NCHW
        jnp.transpose(kernel, [1, 0, 2, 3]),  # to OIHW
        window_strides=(1, 1),
        padding='SAME')
    enh_grad = jax.lax.conv(
        jnp.transpose(enh_pooled, [0, 3, 1, 2]),  # to NCHW
        jnp.transpose(kernel, [1, 0, 2, 3]),  # to OIHW
        window_strides=(1, 1),
        padding='SAME')

    loss_spa = jnp.mean((enh_grad - ori_grad)**2)
    return loss_spa

  def loss_color(enhanced):
    plane_avg = jnp.mean(enhanced, axis=(1, 2), keepdims=True)
    loss_col = jnp.mean((plane_avg[..., 0] - plane_avg[..., 1])**2 +
                        (plane_avg[..., 0] - plane_avg[..., 2])**2 +
                        (plane_avg[..., 1] - plane_avg[..., 2])**2)
    return loss_col

  def loss_exposure(enhanced, rsize=16, E=0.6):  # pylint: disable=invalid-name
    enhanced_gray = jnp.mean(enhanced, axis=-1, keepdims=True)
    avg_intensity = nn.avg_pool(enhanced_gray,
                                window_shape=(rsize, rsize),
                                strides=(rsize, rsize))
    loss_exp = jnp.mean((avg_intensity - E)**2)
    return loss_exp

  def loss_illumination(enhanced):
    delta_h = enhanced[:, 1:, :, :] - enhanced[:, :-1, :, :]
    delta_w = enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :]
    tv = jnp.mean(delta_h**2, axis=(1, 2)) + jnp.mean(delta_w**2, axis=(1, 2))
    loss_tv = jnp.mean(tv.sum(axis=-1))
    return loss_tv

  def loss_fn(params):
    img, alphas = Model(CONFIG.model).apply({'params': params},
                                            batch,
                                            train=True)

    loss_tv = CONFIG.loss_tv_weight * loss_illumination(alphas)
    loss_spa = CONFIG.loss_spa_weight * loss_spatial(batch, img)
    loss_col = CONFIG.loss_col_weight * loss_color(img)
    loss_exp = CONFIG.loss_exp_weight * loss_exposure(img)
    loss = loss_tv + loss_spa + loss_col + loss_exp

    return loss, {
        'loss_tv': loss_tv,
        'loss_spa': loss_spa,
        'loss_col': loss_col,
        'loss_exp': loss_exp,
        'loss': loss
    }

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, losses), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, losses


def train_epoch(state, train_ds, epoch):
  """Trains for a single epoch."""

  batch_metrics = []

  for img_batch in tqdm(train_ds, desc=f'Epoch {epoch}'):
    state, metrics = train_step(state, img_batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics = jax.device_get(batch_metrics)
  epoch_metrics = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0]
  }

  return state, epoch_metrics


def create_train_state(rng):
  """Creates initial `TrainState`."""
  model = Model(CONFIG.model)
  inp_shape = [1, CONFIG.image_size, CONFIG.image_size, 3]
  params = model.init(rng, jnp.ones(inp_shape))['params']
  tx = optax.chain(optax.clip_by_global_norm(max_norm=CONFIG.grad_clip_norm),
                   optax.adamw(CONFIG.lr, weight_decay=CONFIG.weight_decay))
  return train_state.TrainState.create(apply_fn=model.apply,
                                       params=params,
                                       tx=tx)


def save_state(state, epoch, work_dir):
  ckpt_dir = ensure_path(os.path.join(work_dir, 'models'))
  checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                              target=state,
                              step=epoch,
                              keep_every_n_steps=10,
                              overwrite=False)


def setup_env():
  """Environment setup"""

  # disables the preallocation behavior
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  # prevent tensorflow from preallocating memory
  tf.config.experimental.set_visible_devices([], 'GPU')

def main(_):
  setup_env()

  parse_config()
  setup_logger(CONFIG.work_dir)

  train_ds = build_dataset(CONFIG.train_dir, CONFIG.batch_size,
                           CONFIG.image_size)

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)

  state = create_train_state(init_rng)

  for epoch in range(1, CONFIG.num_epochs + 1):
    state, metrics = train_epoch(state, train_ds, epoch)
    logging.info('Train epoch: %d, %s', epoch,
                 ', '.join(f'{k}: {v:.3f}' for k, v in metrics.items()))
    save_state(state, epoch, CONFIG.work_dir)


if __name__ == '__main__':
  app.run(main)
