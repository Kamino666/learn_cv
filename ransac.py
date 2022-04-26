import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from queue import PriorityQueue


class RansacLine:
    p_num = 2  # 最小点集的大小

    @staticmethod
    def step(subset, all_points, threshold):
        # 2. hypothesize the model
        (x0, y0), (x1, y1) = subset
        # Ax+By+C=0
        A = y1 - y0
        B = x0 - x1
        C = y0 * (x1 - x0) - x0 * (y1 - y0)
        # normalize
        normalize_constant = math.sqrt(A ** 2 + B ** 2)
        A, B, C = A / normalize_constant, B / normalize_constant, C / normalize_constant

        # 3. compute the error
        votes_num = 0
        for p in all_points:
            if abs(A * p[0] + B * p[1] + C) <= threshold:
                votes_num += 1
        return votes_num

    @staticmethod
    def draw(subset, dist, x_range=100):
        # Ax+By+C=0
        (y0, x0), (y1, x1) = subset
        A = y1 - y0
        B = x0 - x1
        C = y0 * (x1 - x0) - x0 * (y1 - y0)
        normalize_constant = math.sqrt(A ** 2 + B ** 2)
        A, B, C = A / normalize_constant, B / normalize_constant, C / normalize_constant
        C1, C2 = C + dist, C - dist
        x = np.linspace(0, x_range, 10)
        # compute the line
        ori_line_y = (-C - A * x) / B
        up_line_y = (-C1 - A * x) / B
        down_line_y = (-C2 - A * x) / B
        # draw
        plt.plot(x, ori_line_y, 'k')
        plt.plot(x, up_line_y, 'm--')
        plt.plot(x, down_line_y, 'm--')


class RansacCircle:
    p_num = 3  # 最小点集的大小

    @staticmethod
    def step(subset: list, all_points, threshold):
        (center_x, center_y), radius = RansacCircle._get_circle(subset[0], subset[1], subset[2])
        if radius is None:
            return 0
        # 3. compute the error
        votes_num = 0
        for p in all_points:
            # 点到圆的距离是到圆心的距离减半径的绝对值
            dist = math.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2)
            if radius - threshold <= dist <= radius + threshold:
                votes_num += 1
        return votes_num

    @staticmethod
    def _get_circle(p1, p2, p3):
        """三点求圆，返回圆心和半径，复数求法，假如共线返回半径为None"""
        # 假如三点共线则没有圆
        if abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) <= 1e-6:
            return (0, 0), None
        x, y, z = p1[0] + p1[1] * 1j, p2[0] + p2[1] * 1j, p3[0] + p3[1] * 1j
        w = z - x
        w /= y - x
        c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
        return (-c.real, -c.imag), abs(c + x)

    @staticmethod
    def draw(subset, dist):
        """画圆"""
        (center_y, center_x), radius = RansacCircle._get_circle(subset[0], subset[1], subset[2])
        # draw
        plt.gcf().gca().add_artist(plt.Circle((center_x, center_y), radius, fill=False, color='k'))
        plt.gcf().gca().add_artist(plt.Circle((center_x, center_y), radius - dist, fill=False, ls='--', color='m'))
        plt.gcf().gca().add_artist(plt.Circle((center_x, center_y), radius + dist, fill=False, ls='--', color='m'))


class RansacSquare:
    p_num = 2  # 最小点集的大小

    @staticmethod
    def step(subset, all_points, threshold):
        # 2. hypothesize the model
        # A-----D
        # |  M  |
        # B-----C
        A, C = np.array(subset[0]), np.array(subset[1])
        B, D = RansacSquare.get_the_other_two_point(A, C)
        lines = [RansacSquare.two_p_to_a_line(i[0], i[1]) for i in [(A, B), (A, D), (C, D), (B, C)]]
        # 3. compute the error
        votes_num = 0
        for p in all_points:
            dist = [abs(a * p[0] + b * p[1] + c) for a, b, c in lines]  # 离正方形距离就是离最近边的距离
            if min(dist) <= threshold:
                votes_num += 1
        return votes_num

    @staticmethod
    def draw(subset, dist):
        """绘制正方形和对应边界，因为懒，直接画四条直线"""
        A, C = np.array(subset[0]), np.array(subset[1])
        B, D = RansacSquare.get_the_other_two_point(A, C)
        RansacLine.draw(((A[0], A[1]), (B[0], B[1])), dist)
        RansacLine.draw(((A[0], A[1]), (D[0], D[1])), dist)
        RansacLine.draw(((C[0], C[1]), (B[0], B[1])), dist)
        RansacLine.draw(((C[0], C[1]), (D[0], D[1])), dist)

    @staticmethod
    def two_p_to_a_line(p1, p2):
        """已知两个点，求归一化后的直线方程（距离可以直接|ax+by+c|计算，不用分母）"""
        # ax+by+c=0
        a, b, c = p2[1] - p1[1], p1[0] - p2[0], p1[1] * (p2[0] - p1[0]) - p1[0] * (p2[1] - p1[1])
        # normalize
        normalize_constant = math.sqrt(a ** 2 + b ** 2)
        a, b, c = a / normalize_constant, b / normalize_constant, c / normalize_constant
        return a, b, c

    @staticmethod
    def get_the_other_two_point(A, C):
        """已知一个正方形的对角点，求另外两个点的坐标，返回numpy数组"""
        AM = (C - A) / 2
        MB, MD = AM[::-1].copy(), AM[::-1].copy()
        MB[0] = -MB[0]
        MD[1] = -MD[1]
        B, D = A + AM + MB, A + AM + MD
        return B, D


def ransac(x: np.ndarray, model, k, threshold):
    """
    主要调用的函数
    :param x: 图片
    :param model: 模型，是上面定义的类
    :param k: 迭代次数
    :param threshold: 阈值
    :return: 得票数, 选择的点集
    """
    res = PriorityQueue()
    all_points = list(zip(np.where(x == 1)[0], np.where(x == 1)[1]))
    assert k < len(all_points) * (len(all_points) - 1), "iteration too big"
    visited_subsets = []
    # iterate k steps
    for i in range(k):
        # 1. Randomly select minimal subset of points
        subset = random.sample(all_points, model.p_num)
        while subset in visited_subsets:
            subset = random.sample(all_points, model.p_num)
        visited_subsets.append(subset)
        res.put((-model.step(subset, all_points, threshold), subset))
    return res.get()


# 先简单检测一个直线
img = cv2.imread("imgs/ransac1.png", cv2.IMREAD_GRAYSCALE)
bin_img = img.copy()
bin_img[img == 0] = 1
bin_img[img == 255] = 0
votes, line = ransac(bin_img, RansacLine(), 1000, 1.5)
fig = plt.figure(1)
ax = plt.gca()
ax.imshow(img, cmap=plt.get_cmap('gray'))
RansacLine.draw(line, 1.5)
ax.scatter([i[1] for i in line], [i[0] for i in line], c='red')
ax.annotate(f"votes number: {-votes}", (0, 0))
plt.ylim([img.shape[0], 0])
# plt.show()

# 然后检测一下圆和正方形
img = cv2.imread("imgs/ransac5.png", cv2.IMREAD_GRAYSCALE)
bin_img = img.copy()
bin_img[img == 0] = 1
bin_img[img == 255] = 0
fig = plt.figure(2)
ax = plt.gca()
votes_circle, line_circle = ransac(bin_img, RansacCircle(), 300, 1.5)
votes_square, line_square = ransac(bin_img, RansacSquare(), 1400, 2)
ax.imshow(img, cmap=plt.get_cmap('gray'))
RansacCircle.draw(line_circle, 1.5)
RansacSquare.draw(line_square, 2)
plt.scatter([i[1] for i in line_circle], [i[0] for i in line_circle], c='red')
plt.scatter([i[1] for i in line_square], [i[0] for i in line_square], c='red')
plt.annotate(f"Circle votes: {-votes_circle}. Square votes: {-votes_square}", (0, 0))
plt.ylim([img.shape[0], 0])

plt.show()
