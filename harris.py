import numpy as np
import matplotlib.pyplot as plt
import cv2


# 仍然是简单的非最大化抑制
def non_max_suppression(x):
    res = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == 0:
                continue
            if np.sum(x[i - 1:i + 2, j - 1:j + 2] > x[i, j]) == 0:
                # print(x[i - 1:i + 2, j - 1:j + 2])
                res[i, j] = x[i, j]
    return res


img = cv2.imread("imgs/chess_board.png", cv2.IMREAD_GRAYSCALE)
aperture_size = 3  # 梯度的窗大小
block_size = 3  # 为了代码易懂就默认考虑的邻域是3
alpha = 0.06
r_threshold = 500000
Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=aperture_size)
Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=aperture_size)
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # 高斯核
# kernel = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159

# 循环每一点
R = np.zeros(img.shape)
for i in range(2, img.shape[0] - 2):
    for j in range(2, img.shape[1] - 2):
        ix2 = np.sum(Ix[i - 1:i + 2, j - 1:j + 2] ** 2 * kernel)
        iy2 = np.sum(Iy[i - 1:i + 2, j - 1:j + 2] ** 2 * kernel)
        ixiy = np.sum(Ix[i - 1:i + 2, j - 1:j + 2] * Iy[i - 1:i + 2, j - 1:j + 2] * kernel)
        r = (ix2 * iy2 - ixiy ** 2) - alpha * (ix2 + iy2) ** 2
        R[i, j] = r
        # if r > r_threshold:
        #     R[i, j] = r

# 阈值
# R[R < r_threshold] = 0
# 非最大化抑制
res = non_max_suppression(R)
# res = R
points = np.where(res > res.max()*0.5)

# 画图
plt.subplot(2, 2, 1)
plt.imshow(R)
plt.title("R")
plt.subplot(2, 2, 2)
plt.title("result")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.scatter(points[1], points[0])

# cv2
plt.subplot(2, 2, 3)
plt.title("opencv R")
cv2_res = cv2.cornerHarris(img, 3, 3, alpha)
plt.imshow(cv2_res)
plt.subplot(2, 2, 4)
plt.title("opencv result")
cv2_points = np.where(cv2_res > cv2_res.max()*0.5)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.scatter(cv2_points[1], cv2_points[0])
plt.show()
