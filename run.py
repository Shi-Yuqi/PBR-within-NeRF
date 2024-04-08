# from scene import FallenParticle
# import taichi as ti

# ti.init(arch=ti.gpu)

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
# from tqdm import trange
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
    N_samples: int,                 #采样数量
    perturb: Optional[bool] = True, #扰动设置
    inverse_depth: bool = False     #反向深度
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    t_vals = torch.linspace(0.0, 1.0, N_samples, device=rays_o.device)

    if not inverse_depth:
        z_vals = near * (1.0 - t_vals) + far * t_vals #从远到近线性采样
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals) #反向深度采样

    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat((mids, z_vals[-1:]), dim=-1)
        lower = torch.concat((z_vals[:1], mids), dim=-1)
        t_rand = torch.rand([N_samples],device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

#对采样点进行可视化分析
# y_vals = torch.zeros_like(z_vals)
# # 调用采样策略函数
# _, z_vals_unperturbed = sample_stratified(rays_o, rays_d, near, far, N_samples,
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
  
class NeRF(nn.Module):
   def __init__(
        self,
        d_input: int,
        n_layers
        d_hidden: int = 256,
        )
      
