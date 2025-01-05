# 显式定义一个5x5随机0,1数组 (这里只是一个例子，不是真正的随机生成)
from skimage import measure
import numpy as np
import cv2
# random_binary_array = np.array([
#     [1, 1, 1, 0, 1],
#     [1, 1, 0, 0, 1],
#     [1, 0, 0, 0, 1],
#     [0, 0, 0, 1, 1],
#     [1, 0, 0, 0, 1]
# ])
random_binary_array = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
random_binary_array1 = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
random_binary_array2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
# random_binary_array1 = np.array([
#     [1, 1, 1, 0, 1],
#     [1, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1],
#     [1, 1, 0, 0, 1]
# ])
pred_labeled, pred_num = measure.label(random_binary_array, return_num=True, connectivity=1)
pred_labeled1, pred_num1 = measure.label(random_binary_array1, return_num=True, connectivity=1)

res = cv2.bitwise_or(pred_labeled, random_binary_array2)
gt_res = np.zeros(pred_labeled1.shape, dtype=np.uint8)
# gt_res[pred_labeled1 == 2]=1
unique_values = np.unique(pred_labeled1)
# print(unique_values)
output_array = np.ones_like(pred_labeled1) 
for value in unique_values:
    # 找到 gt_labeled 中等于当前唯一值的位置，并在 output_array 中对应位置设为0
    output_array[pred_labeled1 == value] = 1
print(output_array)