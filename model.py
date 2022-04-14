"""Main file for Zero-DCE model"""

from typing import Any, Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
import jax


def conv3x3(out_dim, mid_dim=None, use_depthwise_conv=False, **kw):
  args = {
      'kernel_size': (3, 3),
      'kernel_init': nn.initializers.normal(stddev=2e-2),
  }
  args.update(kw)

  if out_dim is None:
    mid_dim = out_dim

  if use_depthwise_conv:
    return nn.Sequential([
        nn.Conv(features=mid_dim, feature_group_count=mid_dim, **args),
        nn.Conv(features=out_dim, **args)
    ])
  else:
    return nn.Conv(features=out_dim, **args)


class BasicBlock(nn.Module):
  """Basic layer """
  out_dim: int
  mid_dim: int
  use_depthwise_conv: bool = False
  use_bias: bool = True
  act: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               bridge: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    inp = x if bridge is None else jnp.concatenate([bridge, x], axis=-1)
    out = conv3x3(self.out_dim,
                  self.mid_dim,
                  self.use_depthwise_conv,
                  use_bias=self.use_bias)(inp)
    out = self.act(out)  # pylint: disable=too-many-function-args
    return out


class ZeroDCE(nn.Module):
  """The Zero-DCE model function with curve estimation

  For more model details, check the CVPR paper:
  Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

  Attributes:
    features: initial hidden dimension for the input resolution.
    depth: the number of convolution layers for the model.
    num_iters: number of iterations.
    use_bias: whether to use bias in all the convolution layers.
  """
  features: int = 32
  depth: int = 7
  num_iters: int = 8
  rescale_factor: int = 1
  reuse_alpha_map: bool = False
  use_depthwise_conv: bool = False
  use_bias: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = False) -> Any:

    assert self.depth % 2 == 1
    b, h, w, c = x.shape  # input image shape

    img = x

    if not train and self.rescale_factor > 1:
      new_shape = (b, h // self.rescale_factor, w // self.rescale_factor, c)
      x = jax.image.resize(x, shape=new_shape, method='bilinear')

    bridges = []
    for i in range(self.depth // 2 + 1):
      x = BasicBlock(out_dim=self.features,
                     mid_dim=self.features if i > 0 else c,
                     use_depthwise_conv=self.use_depthwise_conv,
                     use_bias=self.use_bias,
                     name=f'layer_{i}')(x)

      if i < self.depth // 2:
        bridges.append(x)

    for i in range(self.depth // 2 + 1, self.depth - 1):
      x = BasicBlock(out_dim=self.features,
                     mid_dim=self.features,
                     use_depthwise_conv=self.use_depthwise_conv,
                     use_bias=self.use_bias,
                     name=f'layer_{i}')(x, bridges.pop())

    num_alpha_maps = 1 if self.reuse_alpha_map else self.num_iters
    alphas = BasicBlock(out_dim=c * num_alpha_maps,
                        mid_dim=self.features,
                        use_bias=self.use_bias,
                        act=nn.tanh,
                        name='last')(x, bridges.pop())

    if not train and self.rescale_factor > 1:
      c = alphas.shape[-1]
      alphas = jax.image.resize(alphas, (b, h, w, c), method='bilinear')

    if self.reuse_alpha_map:
      alpha_each_iter = [alphas] * self.num_iters
    else:
      alpha_each_iter = jnp.split(alphas, self.num_iters, axis=-1)

    for i in range(self.num_iters):
      img = img + alpha_each_iter[i] * (img**2 - img)

    return img, alphas


def Model(variant='Zero-DCE', **kw):  # pylint: disable=invalid-name
  """Factory function to create a model variant

  Every model file should have this Model() function that returns the flax
  model function. The function name should be fixed.

  Args:
    variant: Zero-DCE model variants. Options: 'Zero-DCE' | 'Zero-DCE++'
    **kw: Other config dicts.
  """

  config = {
      # params: 0.079 M, GFlops: 84.990
      'Zero-DCE': {
          'features': 32,
          'depth': 7,
          'num_iters': 8,
          'rescale_factor': 1,
          'reuse_alpha_map': False,
          'use_depthwise_conv': False,
          'use_bias': True
      },
      # params: 0.010 M, GFlops: 0.115
      'Zero-DCE++': {
          'features': 32,
          'depth': 7,
          'num_iters': 8,
          'rescale_factor': 12,
          'reuse_alpha_map': True,
          'use_depthwise_conv': True,
          'use_bias': True
      },
  }[variant]

  for k, v in config.items():
    kw.setdefault(k, v)

  return ZeroDCE(**kw)
