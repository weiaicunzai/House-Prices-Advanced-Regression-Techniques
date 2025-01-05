import torch
import cv2
import numpy as np

# 创建一个简单的 2D 输入张量 (batch_size=1, channels=1, height, width)
img_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.float32)

# 将 PyTorch 张量转换为 NumPy 数组，并移除批次和通道维度，同时转换为 OpenCV 支持的格式
img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)

# 定义一个 3x3 的结构元素（全为1）
kernel = np.ones((3, 3), np.uint8)

# 执行收缩操作
eroded_img_np = cv2.erode(img_np, kernel, iterations=1)

# 将结果转换回 PyTorch 张量，并归一化到 [0, 1] 范围，恢复批次和通道维度
eroded_img_tensor = torch.from_numpy(eroded_img_np.astype(np.float32) / 255.0)[None, None]

print("Original Padded Image Tensor:")
print(img_tensor.squeeze())  # 移除大小为1的维度以便更清晰地打印

print("\nEroded Image Tensor using opencv-python:")
print(eroded_img_tensor.squeeze())