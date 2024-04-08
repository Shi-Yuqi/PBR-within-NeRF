d_input = 3           # 输入维度
n_freqs = 10          # 输入到编码函数中的样本点数量
log_space = True      # 如果设置，频率按对数空间缩放
use_viewdirs = True   # 如果设置，则使用视图方向作为输入
n_freqs_views = 4     # 视图编码功能的数量

# 采样策略
n_samples = 64         # 每条射线 的空间样本数
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