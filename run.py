# from scene import FallenParticle
# import taichi as ti

# ti.init(arch=ti.gpu)

d_input = 3           # 输入维度
n_freqs = 10          # 输入到编码函数中的样本点数量
log_space = True      # 如果设置，频率按对数空间缩放
use_viewdirs = True   # 如果设置，则使用视图方向作为输入
n_freqs_views = 4     # 视图编码功能的数量

# 采样策略
n_samples = 64         # 每条射线的空间样本数
perturb = True         # 如果设置，则对采样位置应用噪声
inverse_depth = False  # 如果设置，则按反深度线性采样点

# 模型
d_filter = 128          # 线性层滤波器的尺寸
n_layers = 2            # bottleneck层数量
skip = []               # 应用输入残差的层级
use_fine_model = True   # 如果设置，则创建一个精细模型
d_filter_fine = 128     # 精细网络线性层滤波器的尺寸
n_layers_fine = 6       # 精细网络瓶颈层数

# 分层采样
n_samples_hierarchical = 64   # 每条射线的样本数
perturb_hierarchical = False  # 如果设置，则对采样位置应用噪声

# 优化器
lr = 5e-4  # 学习率

# 训练
n_iters = 10000
batch_size = 2**14          # 每个梯度步长的射线数量（2 的幂次）
one_image_per_step = True   # 每个梯度步骤一个图像（禁用批处理）
chunksize = 2**14           # 根据需要进行修改，以适应 GPU 内存
center_crop = True          # 裁剪图像的中心部分（每幅图像裁剪一次）
center_crop_iters = 50      # 经过这么多epoch后，停止裁剪中心
display_rate = 25          # 每 X 个epoch显示一次测试输出

# 早停
warmup_iters = 100          # 热身阶段的迭代次数
warmup_min_fitness = 10.0   # 在热身_iters 处继续训练的最小 PSNR 值
n_restarts = 10             # 训练停滞时重新开始的次数

# 捆绑了各种函数的参数，以便一次性传递。
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}
kwargs_sample_hierarchical = {
    'perturb': perturb
}

# def main():
#     n_particles = 10
#     dt = 1e-4
#     gravity = [0, -98.1]
#     circle_radius = 10
#     window_res = (512, 512)

#     scene = FallenParticle(
#         n_particles=n_particles,
#         dt=dt,
#         gravity=gravity,
#         circle_radius=circle_radius,
#         window_res=window_res,
#     )
#     scene.run_simulation()

# if __name__ == "__main__":
#     main()

import os
from typing import Optional,Tuple,List,Union,Callable

import cv2
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
from photoextractor import PhotoExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_matrices = np.load('D:\\code\\data_PBR_within_NeRF\\matrix\\matrices.npy')
# loaded_matrices = np.load('D:\\code\\data_PBR_within_NeRF\\matrix_trans\\matrices.npy')
# loaded_matrices = np.load('E:\\matrix\\matrices.npy')

path_photo = 'D:\\code\\data_PBR_within_NeRF\\within_texture'
# path_photo = 'E:\\blender\\within_texture'
data = PhotoExtractor(path_photo, 20)
images = np.array(data.extract_photos())  # 图像数据

plt.imshow(images[0])  
plt.show() 

poses = loaded_matrices  # 位姿数据
focal = np.full(1, 30)  # 焦距数值

print(f'Images shape: {images.shape}')
print(f'Poses shape: {poses.shape}')
print(f'Focal length: {focal}')

height, width = (1080,1920)
near, far = 2., 6.

n_training = 30 # 训练数据数量
testimg_idx = 31 # 测试数据下标
testimg, testpose = images[testimg_idx], poses[testimg_idx]


plt.imshow(testimg)
print('Pose')
print(testpose)

# 方向数据
dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
# 原点数据
origins = poses[:, :3, -1]


# 绘图的设置

ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
_ = ax.quiver(
  origins[..., 0].flatten(),
  origins[..., 1].flatten(),
  origins[..., 2].flatten(),
  dirs[..., 0].flatten(),
  dirs[..., 1].flatten(),
  dirs[..., 2].flatten(), length=0.5, normalize=True)

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('z')

plt.show()

def get_rays(
        H:int,
        W:int,
        focal:float,
        pose:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    #启用针孔相机收集每个像素的方向
    x, y = torch.meshgrid(
        torch.arange(W, dtype=torch.float32).to(pose),
        torch.arange(H, dtype=torch.float32).to(pose),
        indexing="xy"
    )

    x = x.transpose(-1, -2)
    y = y.transpose(-1, -2)

    directions = torch.stack([(x - W / 2) / focal, -(y - H / 2) / focal, -torch.ones_like(x)], dim=-1)

    rays_d = directions @ pose[:3, :3].T
    rays_d = torch.sum(directions[..., None, :] * pose[:3, :3], -1)

    rays_o = pose[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d
    
testimg = torch.from_numpy(images[testimg_idx]).to(device)
testpose = torch.from_numpy(poses[testimg_idx]).to(device)

images = torch.from_numpy(images[:n_training]).to(device)
poses = torch.from_numpy(poses).to(device)
focal = torch.from_numpy(focal).to(device)

# testimg = torch.from_numpy(images[testimg_idx]).to(device)
# testpose = torch.from_numpy(poses[testimg_idx]).to(device)

H, W = images.shape[1:3]

with torch.no_grad():
    rays_o, rays_d = get_rays(H, W, focal, testpose)

print("ray origin")
print(rays_o.shape)

print(rays_o[H//2, W//2, :])
print("")

print("ray direction")

print(rays_d.shape)
print(rays_d[H//2, W//2, :])
print("")

def sample_stratified(
    rays_o: torch.Tensor,           #射线原点
    rays_d: torch.Tensor,           #射线方向
    near: float,
    far: float,
    n_samples: int,                 #采样数量
    perturb: Optional[bool] = True, #扰动设置
    inverse_depth: bool = False     #反向深度
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)

    if not inverse_depth:
        z_vals = near * (1.0 - t_vals) + far * t_vals #从远到近线性采样
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals) #反向深度采样

    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat((mids, z_vals[-1:]), dim=-1)
        lower = torch.concat((z_vals[:1], mids), dim=-1)
        t_rand = torch.rand([n_samples],device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

#对采样点进行可视化分析
# y_vals = torch.zeros_like(z_vals)
# # 调用采样策略函数
# _, z_vals_unperturbed = sample_stratified(rays_o, rays_d, near, far, n_samples,
#                                   perturb=False, inverse_depth=inverse_depth)
# # 绘图相关
# plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), 'b-o')
# plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), 'r-o')
# plt.ylim([-1, 2])
# plt.title('Stratified Sampling (blue) with Perturbation (red)')
# ax = plt.gca()
# ax.axes.yaxis.set_visible(False)
# plt.grid(True)
class PositionalEncoder(nn.Module):
  """
  对输入点做sine或者consine位置编码。
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # 定义线性或者log尺度的频率
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # 替换sin和cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(
    self,
    x
  ) -> torch.Tensor:
    """
    实际使用位置编码的函数。
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
  
# 定义NeRF模型
class NeRF(nn.Module):
  """
  神经辐射场模块。
  """
  def __init__(
    self,
    d_input: int = 3,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional[int] = None
  ):
    super().__init__()
    self.d_input = d_input # 输入
    self.skip = skip # 残差连接
    self.act = nn.functional.relu # 激活函数
    self.d_viewdirs = d_viewdirs # 视图方向

    # 创建模型的层结构
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck 层
    if self.d_viewdirs is not None:
      # 如果使用视图方向，分离alpha和RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # 如果不使用试图方向，则简单输出
      self.output = nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    r"""
    带有视图方向的前向传播
    """

    # 判断是否设置视图方向
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # 运行bottleneck层之前的网络层
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # 运行 bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # 结果传入到rgb过滤器
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)

      # 拼接alpha一起作为输出
      x = torch.concat([x, alpha], dim=-1)
    else:
      # 不拼接，简单输出
      x = self.output(x)
    return x
  
  # 体积渲染
def cumprod_exclusive(
  tensor: torch.Tensor
) -> torch.Tensor:
  """
  (Courtesy of https://github.com/krrish94/nerf-pytorch)

  和tf.math.cumprod(..., exclusive=True)功能类似

  参数:
  tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
    is to be computed.
  返回值:
  cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
    tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
  """

  # 首先计算规则的cunprod
  cumprod = torch.cumprod(tensor, -1)
  cumprod = torch.roll(cumprod, 1, -1)
  # 用1替换首个元素
  cumprod[..., 0] = 1.
  
  return cumprod

# 输出到图像的函数
def raw2outputs(
  raw: torch.Tensor,
  z_vals: torch.Tensor,
  rays_d: torch.Tensor,
  raw_noise_std: float = 0.0,
  white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  将NeRF的输出转换为RGB输出。
  """

  # 沿着`z_vals`轴元素之间的差值.
  dists = z_vals[..., 1:] - z_vals[..., :-1]
  dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

  # 将每个距离乘以相应方向射线的法线，转换为现实世界中的距离（考虑非单位方向）。
  dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

  # 为模型预测密度添加噪音。可用于在训练过程中对网络进行正则化（防止出现浮点伪影）。
  noise = 0.
  if raw_noise_std > 0.:
    noise = torch.randn(raw[..., 3].shape) * raw_noise_std

  # Predict density of each sample along each ray. Higher values imply
  # higher likelihood of being absorbed at this point. [n_rays, n_samples]
  alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

  # 预测每条射线上每个样本的密度。数值越大，表示该点被吸收的可能性越大。[n_ 射线，n_样本］
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

  # 计算RGB图的权重。
  rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
  rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

  # 估计预测距离的深度图。
  depth_map = torch.sum(weights * z_vals, dim=-1)

  # 稀疏图
  disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))

  # 沿着每条射线加权。
  acc_map = torch.sum(weights, dim=-1)

  # 要合成到白色背景上，请使用累积的 alpha 贴图。
  if white_bkgd:
    rgb_map = rgb_map + (1. - acc_map[..., None])

  return rgb_map, depth_map, acc_map, weights

# 采样概率密度函数
def sample_pdf(
  bins: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> torch.Tensor:
  """
  应用反向转换采样到一组加权点。
  """

  # 正则化权重得到概率密度函数。
  pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

  # 将概率密度函数转为累计分布函数。
  cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
  cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

  # 从累计分布函数中提取样本位置。perturb == 0 时为线性。
  if not perturb:
    u = torch.linspace(0., 1., n_samples, device=cdf.device)
    u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
  else:
    u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

  # 沿累计分布函数找出 u 值所在的索引。
  u = u.contiguous() # 返回具有相同值的连续张量。
  inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

  # 夹住超出范围的索引。
  below = torch.clamp(inds - 1, min=0)
  above = torch.clamp(inds, max=cdf.shape[-1] - 1)
  inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

  # 从累计分布函数和相应的 bin 中心取样。
  matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
  cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                       index=inds_g)
  bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

  # 将样本转换为射线长度。
  denom = (cdf_g[..., 1] - cdf_g[..., 0])
  denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
  t = (u - cdf_g[..., 0]) / denom
  samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

  return samples # [n_rays, n_samples]

def get_chunks(
  inputs: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  """
  输入分块。
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_chunks(
  points: torch.Tensor,
  encoding_function: Callable[[torch.Tensor], torch.Tensor],
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  """
  对点进行编码和分块，为 NeRF 模型做好准备。
  """
  points = points.reshape((-1, 3))
  points = encoding_function(points)
  points = get_chunks(points, chunksize=chunksize)
  return points

def prepare_viewdirs_chunks(
  points: torch.Tensor,
  rays_d: torch.Tensor,
  encoding_function: Callable[[torch.Tensor], torch.Tensor],
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  r"""
  对视图方向进行编码和分块，为 NeRF 模型做好准备。
  """
  viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
  viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
  viewdirs = encoding_function(viewdirs)
  viewdirs = get_chunks(viewdirs, chunksize=chunksize)
  return viewdirs
def nerf_forward(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  encoding_fn: Callable[[torch.Tensor], torch.Tensor],
  coarse_model: nn.Module,
  kwargs_sample_stratified: dict = None,
  n_samples_hierarchical: int = 0,
  kwargs_sample_hierarchical: dict = None,
  fine_model = None,
  viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
  chunksize: int = 2**15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
  """
  计算一次前向传播
  """

  # 设置参数
  if kwargs_sample_stratified is None:
    kwargs_sample_stratified = {}
  if kwargs_sample_hierarchical is None:
    kwargs_sample_hierarchical = {}
  
  # 沿着每条射线的样本查询点。
  query_points, z_vals = sample_stratified(
      rays_o, rays_d, near, far, **kwargs_sample_stratified)

  # 准备批次。
  batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
  if viewdirs_encoding_fn is not None:
    batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                               viewdirs_encoding_fn,
                                               chunksize=chunksize)
  else:
    batches_viewdirs = [None] * len(batches)

  # 稀疏模型流程。
  predictions = []
  for batch, batch_viewdirs in zip(batches, batches_viewdirs):
    predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
  raw = torch.cat(predictions, dim=0)
  raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

  # 执行可微分体积渲染，重新合成 RGB 图像。
  rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
  
  outputs = {
      'z_vals_stratified': z_vals
  }

  if n_samples_hierarchical > 0:
    # Save previous outputs to return.
    rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

    # 对精细查询点进行分层抽样。
    query_points, z_vals_combined, z_hierarch = sample_hierarchical(
      rays_o, rays_d, z_vals, weights, n_samples_hierarchical,**kwargs_sample_hierarchical)

    # 像以前一样准备输入。
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
      batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                 viewdirs_encoding_fn,
                                                 chunksize=chunksize)
    else:
      batches_viewdirs = [None] * len(batches)

    # 通过精细模型向前传递新样本。
    fine_model = fine_model if fine_model is not None else coarse_model
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
      predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # 执行可微分体积渲染，重新合成 RGB 图像。
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)
    
    # 存储输出
    outputs['z_vals_hierarchical'] = z_hierarch
    outputs['rgb_map_0'] = rgb_map_0
    outputs['depth_map_0'] = depth_map_0
    outputs['acc_map_0'] = acc_map_0

  # 存储输出
  outputs['rgb_map'] = rgb_map
  outputs['depth_map'] = depth_map
  outputs['acc_map'] = acc_map
  outputs['weights'] = weights
  return outputs

# 绘制采样函数
def plot_samples(
  z_vals: torch.Tensor,
  z_hierarch: Optional[torch.Tensor] = None,
  ax: Optional[np.ndarray] = None):
  r"""
  绘制分层样本和（可选）分级样本。
  """
  y_vals = 1 + np.zeros_like(z_vals)

  if ax is None:
    ax = plt.subplot()
  ax.plot(z_vals, y_vals, 'b-o')
  if z_hierarch is not None:
    y_hierarch = np.zeros_like(z_hierarch)
    ax.plot(z_hierarch, y_hierarch, 'r-o')
  ax.set_ylim([-1, 2])
  ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
  ax.axes.yaxis.set_visible(False)
  ax.grid(True)
  return ax

def crop_center(
  img: torch.Tensor,
  frac: float = 0.5
) -> torch.Tensor:
  r"""
  从图像中裁剪中心方形。
  """
  h_offset = round(img.shape[0] * (frac / 2))
  w_offset = round(img.shape[1] * (frac / 2))
  return img[h_offset:-h_offset, w_offset:-w_offset]

class EarlyStopping:
  r"""
  基于适配标准的早期停止辅助器
  """
  def __init__(
    self,
    patience: int = 30,
    margin: float = 1e-4
  ):
    self.best_fitness = 0.0
    self.best_iter = 0
    self.margin = margin
    self.patience = patience or float('inf')  # 在epoch停止提高后等待的停止时间

  def __call__(
    self,
    iter: int,
    fitness: float
  ):
    r"""
    检查是否符合停止标准。
    """
    if (fitness - self.best_fitness) > self.margin:
      self.best_iter = iter
      self.best_fitness = fitness
    delta = iter - self.best_iter
    stop = delta >= self.patience  # 超过耐性则停止训练
    return stop
  
def init_models():
  r"""
  为 NeRF 训练初始化模型、编码器和优化器。
  """
  # 编码器
  encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
  encode = lambda x: encoder(x)

  # 视图方向编码
  if use_viewdirs:
    encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                        log_space=log_space)
    encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encoder_viewdirs.d_output
  else:
    encode_viewdirs = None
    d_viewdirs = None

  # 模型
  model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
              d_viewdirs=d_viewdirs)
  model.to(device)
  model_params = list(model.parameters())
  if use_fine_model:
    fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
                      d_viewdirs=d_viewdirs)
    fine_model.to(device)
    model_params = model_params + list(fine_model.parameters())
  else:
    fine_model = None

  # 优化器
  optimizer = torch.optim.Adam(model_params, lr=lr)

  # 早停
  warmup_stopper = EarlyStopping(patience=50)

  return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper

model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()

def train():
  r"""
  启动 NeRF 训练。
  """
  # 对所有图像进行射线洗牌。
  if not one_image_per_step:
    height, width = images.shape[1:3]
    all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p), 0)
                        for p in poses[:n_training]], 0)
    rays_rgb = torch.cat([all_rays, images[:, None]], 1)
    rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = rays_rgb.reshape([-1, 3, 3])
    rays_rgb = rays_rgb.type(torch.float32)
    rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
    i_batch = 0

  train_psnrs = []
  val_psnrs = []
  iternums = []
  for i in trange(n_iters):
    model.train()

    if one_image_per_step:
      # 随机选择一张图片作为目标。
      target_img_idx = np.random.randint(images.shape[0])
      target_img = images[target_img_idx].to(device)
      if center_crop and i < center_crop_iters:
        target_img = crop_center(target_img)
      height, width = target_img.shape[:2]
      target_pose = poses[target_img_idx].to(device)
      rays_o, rays_d = get_rays(height, width, focal, target_pose)
      rays_o = rays_o.reshape([-1, 3])
      rays_d = rays_d.reshape([-1, 3])
    else:
      # 在所有图像上随机显示。
      batch = rays_rgb[i_batch:i_batch + batch_size]
      batch = torch.transpose(batch, 0, 1)
      rays_o, rays_d, target_img = batch
      height, width = target_img.shape[:2]
      i_batch += batch_size
      # 一个epoch后洗牌
      if i_batch >= rays_rgb.shape[0]:
          rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
          i_batch = 0
    target_img = target_img.reshape([-1, 3])

    # 运行 TinyNeRF 的一次迭代，得到渲染后的 RGB 图像。
    outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=n_samples_hierarchical,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=chunksize)
    
    # 检查任何数字问题。
    for k, v in outputs.items():
      if torch.isnan(v).any():
        print(f"! [Numerical Alert] {k} contains NaN.")
      if torch.isinf(v).any():
        print(f"! [Numerical Alert] {k} contains Inf.")

    # 反向传播
    rgb_predicted = outputs['rgb_map']
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = -10. * torch.log10(loss)
    train_psnrs.append(psnr.item())

    # 以给定的显示速率评估测试值。
    if i % display_rate == 0:
      model.eval()
      height, width = testimg.shape[:2]
      rays_o, rays_d = get_rays(height, width, focal, testpose)
      rays_o = rays_o.reshape([-1, 3])
      rays_d = rays_d.reshape([-1, 3])
      outputs = nerf_forward(rays_o, rays_d,
                             near, far, encode, model,
                             kwargs_sample_stratified=kwargs_sample_stratified,
                             n_samples_hierarchical=n_samples_hierarchical,
                             kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                             fine_model=fine_model,
                             viewdirs_encoding_fn=encode_viewdirs,
                             chunksize=chunksize)

      rgb_predicted = outputs['rgb_map']
      loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
      print("Loss:", loss.item())
      val_psnr = -10. * torch.log10(loss)
      val_psnrs.append(val_psnr.item())
      iternums.append(i)

      # 绘制输出示例
      fig, ax = plt.subplots(1, 4, figsize=(24,4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
      ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
      ax[0].set_title(f'Iteration: {i}')
      ax[1].imshow(testimg.detach().cpu().numpy())
      ax[1].set_title(f'Target')
      ax[2].plot(range(0, i + 1), train_psnrs, 'r')
      ax[2].plot(iternums, val_psnrs, 'b')
      ax[2].set_title('PSNR (train=red, val=blue')
      z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
      z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
      if 'z_vals_hierarchical' in outputs:
        z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
        z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
      else:
        z_sample_hierarch = None
      _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
      ax[3].margins(0)
      plt.show()

    # 检查 PSNR 是否存在问题，如果发现问题，则停止运行。
    if i == warmup_iters - 1:
      if val_psnr < warmup_min_fitness:
        print(f'Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
        return False, train_psnrs, val_psnrs
    elif i < warmup_iters:
      if warmup_stopper is not None and warmup_stopper(i, psnr):
        print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
        return False, train_psnrs, val_psnrs
    
  return True, train_psnrs, val_psnrs

train()