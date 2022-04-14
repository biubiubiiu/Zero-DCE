"""Utility functions.

NOTE: portions of these codes are taken from
https://github.com/google-research/maxim
"""

import logging
import pathlib
from time import localtime, strftime

import jax.numpy as jnp
import numpy as np
from PIL import Image


def ensure_path(path):
  """Create a leaf directory and all intermediate ones if
  it doest not exists.

  Args:
    path (str): The target directory.
  """
  if isinstance(path, str):
    path = pathlib.Path(path)

  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def setup_logger(log_dir, console_quiet=False):
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.INFO)

  formatter = logging.Formatter('%(asctime)s| %(message)s', '%m-%d|%H:%M:%S')

  if isinstance(log_dir, str):
    log_dir = pathlib.Path(log_dir)

  log_name = strftime('%Y%m%d_%H%M%S', localtime())
  log_name += '.log'
  log_path = ensure_path(log_dir.joinpath('logs', log_name))

  file_hdl = logging.FileHandler(log_path)
  file_hdl.setFormatter(formatter)

  root_logger.addHandler(file_hdl)

  if not console_quiet:
    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(formatter)
    root_logger.addHandler(console_hdl)


def _convert_input_type_range(img):
  """Convert the type and range of the input image.

  It converts the input image to np.float32 type and range of [0, 1].
  It is mainly used for pre-processing the input image in colorspace
  convertion functions such as rgb2ycbcr and ycbcr2rgb.
  Args:
    img (ndarray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float32 type with range [0, 1].
  Returns:
      (ndarray): The converted image with type of np.float32 and range of
          [0, 1].
  """
  img_type = img.dtype
  img = img.astype(np.float32)
  if img_type == np.float32:
    pass
  elif img_type == np.uint8:
    img /= 255.
  else:
    raise TypeError('The img type should be np.float32 or np.uint8, '
                    f'but got {img_type}')
  return img


def _convert_output_type_range(img, dst_type):
  """Convert the type and range of the image according to dst_type.

  It converts the image to desired type and range. If `dst_type` is np.uint8,
  images will be converted to np.uint8 type with range [0, 255]. If
  `dst_type` is np.float32, it converts the image to np.float32 type with
  range [0, 1].
  It is mainly used for post-processing images in colorspace convertion
  functions such as rgb2ycbcr and ycbcr2rgb.
  Args:
    img (ndarray): The image to be converted with np.float32 type and
        range [0, 255].
    dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
        converts the image to np.uint8 type with range [0, 255]. If
        dst_type is np.float32, it converts the image to np.float32 type
        with range [0, 1].
  Returns:
    (ndarray): The converted image with desired type and range.
  """
  if dst_type not in (np.uint8, np.float32):
    raise TypeError('The dst_type should be np.float32 or np.uint8, '
                    f'but got {dst_type}')
  if dst_type == np.uint8:
    img = img.round()
  else:
    img /= 255.

  return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
  """Convert a RGB image to YCbCr image.

  This function produces the same results as Matlab's `rgb2ycbcr` function.
  It implements the ITU-R BT.601 conversion for standard-definition
  television. See more details in
  https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
  It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
  In OpenCV, it implements a JPEG conversion. See more details in
  https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

  Args:
    img (ndarray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float32 type with range [0, 1].
    y_only (bool): Whether to only return Y channel. Default: False.
  Returns:
    ndarray: The converted YCbCr image. The output image has the same type
        and range as input image.
  """
  img_type = img.dtype
  img = _convert_input_type_range(img)
  if y_only:
    out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
  else:
    out_img = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) + [16, 128, 128]
  out_img = _convert_output_type_range(out_img, img_type)
  return out_img


def to_y_channel(img):
  """Change to Y channel of YCbCr.

  Args:
    img (ndarray): Images with range [0, 255].
  Returns:
    (ndarray): Images with range [0, 255] (float type) without round.
  """
  img = img.astype(np.float32) / 255.
  if img.ndim == 3 and img.shape[2] == 3:
    img = rgb2ycbcr(img, y_only=True)
    img = img[..., None]
  return img * 255.


def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False):
  """Calculate PSNR (Peak Signal-to-Noise Ratio).

  Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
  Args:
    img1 (ndarray): Images with range [0, 255].
    img2 (ndarray): Images with range [0, 255].
    crop_border (int): Cropped pixels in each edge of an image. These
        pixels are not involved in the PSNR calculation.
    test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
  Returns:
    float: psnr result.
  """
  assert img1.shape == img2.shape, (
      f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)

  if crop_border != 0:
    img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
    img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

  if test_y_channel:
    img1 = to_y_channel(img1)
    img2 = to_y_channel(img2)

  mse = np.mean((img1 - img2)**2)
  if mse == 0:
    return float('inf')
  return 20. * np.log10(255. / np.sqrt(mse))


def is_image_file(filename):
  """Check if it is an valid image file by suffixs."""
  img_suffixs = [
      '.jpeg', '.JPEG', '.jpg', '.png', '.JPG', '.PNG', '.bmp', '.BMP'
  ]
  if isinstance(filename, str):
    filename = pathlib.Path(filename)

  return filename.suffix in img_suffixs


def save_img(img, pth):
  """Save an image to disk.

  Args:
    img: jnp.ndarry, [height, width, channels], img will be clipped to [0, 1]
      before saved to pth.
    pth: string, path to save the image to.
  """
  Image.fromarray(np.array(
      (np.clip(img, 0., 1.) * 255.).astype(jnp.uint8))).save(pth, 'PNG')
