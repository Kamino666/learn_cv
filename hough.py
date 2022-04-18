"""
coded by Kamino, kamino.plus@qq.com, 2022/4/15
未经许可不得转载
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from queue import PriorityQueue


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


def hough_line(x, theta_prec=1.0, rho_prec=2.0, topk=1, weight_mat: np.ndarray = None, non_max=False):
    if weight_mat is not None:  # 权重矩阵得是方阵，且边长要是单数
        assert len(weight_mat.shape) and weight_mat.shape[0] == weight_mat.shape[1] and weight_mat.shape[0] % 2 == 1
    # theta_ax和rho_ax是霍夫空间两个轴的量化数组
    # 使用theta_prec和rho_prec两个参数来确定量化精度，数字越大量化越粗
    theta_ax = np.deg2rad(np.arange(0, 180, theta_prec))
    max_rho = np.sqrt(x.shape[0] ** 2 + x.shape[1] ** 2)
    rho_ax = np.arange(-max_rho, max_rho, rho_prec)
    # 霍夫空间：是一个rho行theta列的空间
    hough_space = np.zeros((len(rho_ax), len(theta_ax)))
    for i in tqdm(range(x.shape[0])):  # i行 j列
        for j in range(x.shape[1]):
            # 找到二值图像中的1点
            if x[i, j] == 0:
                continue
            # 对于每一个theta的量化值求rho
            for t in range(len(theta_ax)):
                rh = j * np.cos(theta_ax[t]) + i * np.sin(theta_ax[t])
                # 量化得到的rho值，由于rho可能为负数，而数组是没有负数索引的
                # （虽然python的负索引有倒数的意义），所以我们要平移到正确的
                # 位置上。
                # 1.加轴的最大值 2.除以精度 3.找到最近整数位
                # *4.给相邻区块加权添加
                # *5.非最大化抑制
                if weight_mat is None:
                    hough_space[int(round((rh + max_rho) / rho_prec)), t] += 1
                else:
                    tgt_point = (int(round((rh + max_rho) / rho_prec)), t)
                    offset = (weight_mat.shape[0] - 1) // 2
                    hot_area = hough_space[tgt_point[0] - offset:tgt_point[0] + offset + 1,
                               tgt_point[1] - offset:tgt_point[1] + offset + 1]
                    if hot_area.shape == weight_mat.shape:
                        hough_space[tgt_point[0] - offset:tgt_point[0] + offset + 1,
                        tgt_point[1] - offset:tgt_point[1] + offset + 1] += weight_mat
    if non_max is True:
        hough_space = non_max_suppression(hough_space)

    # 用优先队列（大顶堆）找到前n个结果
    if topk == 1:
        points = np.where(hough_space == np.max(hough_space))
        return (rho_ax[points[0]], theta_ax[points[1]]), hough_space
    else:
        queue = PriorityQueue()
        for i in range(hough_space.shape[0]):
            for j in range(hough_space.shape[1]):
                queue.put((-hough_space[i, j], rho_ax[i], theta_ax[j], i, j))
        res = [queue.get()[1:] for _ in range(topk)]
        return res, hough_space


img = cv2.imread("imgs/ironnet_small.jpg", cv2.IMREAD_GRAYSCALE)
canny_img = cv2.Canny(img, 200, 230)
kernel = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159
rho_thetas, hough_img = hough_line(canny_img, topk=10, rho_prec=1, theta_prec=0.5, weight_mat=None, non_max=True)

# 下面都是绘图了
# figure1是对比
fig = plt.figure(1)
ax = plt.subplot(2, 1, 1)
ax.imshow(img, cmap=plt.get_cmap('gray'))
ax.set_title('Result')
print("*" * 30)
print("预测出的直线方程")
for rho, theta, _, _ in rho_thetas:
    print(f"x*{round(np.cos(theta), 2)}+y*{round(np.sin(theta), 2)}={round(rho / 2, 2)}")
    x_ = np.arange(0, canny_img.shape[1])
    y_ = (rho - x_ * np.cos(theta)) / np.sin(theta)
    ax.plot(x_, y_, 'red')
plt.ylim([img.shape[0], 0])
print("*" * 30)
ax = plt.subplot(2, 1, 2)
ax.imshow(canny_img, cmap=plt.get_cmap('gray'))
ax.set_title('Canny')
# figure2是霍夫空间可视化
fig = plt.figure(2)
ax = plt.subplot(1, 1, 1)
ax.imshow(hough_img, origin='lower')
for _, _, ri, ti in rho_thetas:
    ax.scatter(ti, ri)
ax.set_ylabel(r'$\rho$')
ax.set_xlabel(r'$\theta$')
ax.set_title('Hough space')
plt.show()
