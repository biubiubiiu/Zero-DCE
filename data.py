"""Image Dataset"""

import pathlib
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

from utils import is_image_file


def preprocess_image(image, image_size):
  image = tf.image.decode_image(image, channels=3, expand_animations=False)
  image = tf.image.resize(image, [image_size, image_size], antialias=True)
  image /= 255.0  # normalize to [0,1] range
  return image


def load_and_preprocess_image(path, image_size):
  image = tf.io.read_file(path)
  return preprocess_image(image, image_size)


def build_dataset(data_root, batch_size, image_size, shuffle=True):
  if isinstance(data_root, str):
    data_root = pathlib.Path(data_root)

  all_image_paths = [
      p.resolve() for p in data_root.glob('**/*') if is_image_file(p)
  ]
  all_image_paths = [str(path) for path in all_image_paths]

  path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
  ds = path_ds.map(partial(load_and_preprocess_image, image_size=image_size),
                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if shuffle:
    image_count = len(all_image_paths)
    ds = ds.shuffle(buffer_size=image_count)

  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = tfds.as_numpy(ds)
  return ds
