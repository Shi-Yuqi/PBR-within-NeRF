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
# import torch
# from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# from tqdm import trange
from photoextractor import PhotoExtractor

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_matrices = np.load('E:\\matrix\\matrices.npy')

path_photo = 'E:\\blender\\within_texture'
data = PhotoExtractor(path_photo, 20)
images = data.extract_photos()  # 图像数据

plt.imshow(images[0])  
plt.show() 

poses = loaded_matrices  # 位姿数据
focal = 30  # 焦距数值

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
