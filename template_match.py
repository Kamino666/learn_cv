import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import cv2
from PIL import Image
from torchvision.transforms.transforms import ColorJitter, RandomCrop, Compose

img = Image.open("imgs/lena.png")
# 定义不同的模版选择方式
# 1. 随机Crop一小块区域
tf1 = RandomCrop((20, 20))
# 2. 随机Crop稍微大一点的一块区域
tf2 = RandomCrop((40, 40))
# 3. 随机Crop更大一点的一块区域
tf3 = RandomCrop((80, 80))
# 4. Crop + 微弱色彩抖动
tf4 = Compose([
    RandomCrop((80, 80)),
    ColorJitter(brightness=.1, contrast=.2)
])
# 5. Crop + 强烈色彩抖动
tf5 = Compose([
    RandomCrop((80, 80)),
    ColorJitter(brightness=.5, contrast=.5, saturation=.3)
])
tf_list = [tf1, tf2, tf3, tf4, tf5]
methods = [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR]
methods_name = ["CCORR_NORMED", "CCOEFF_NORMED", "CCOEFF", "CCORR"]
col, row = len(tf_list), 2 + len(methods)
for i, tf in enumerate(tf_list):
    template = tf(img)
    img_np, temp_np = np.array(img), np.array(template)
    h, w, _ = temp_np.shape
    plt.subplot(col, row, i * row + 1)
    plt.imshow(temp_np)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(col, row, i * row + 2)
    plt.imshow(img_np)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    for j, method in enumerate(methods):
        ret = cv2.matchTemplate(img_np, temp_np, method=method)
        _, _, _, max_loc = cv2.minMaxLoc(ret)
        ax = plt.subplot(col, row, i * row + 2 + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img_np)
        ax.add_patch(pch.Rectangle(xy=max_loc, width=w, height=h, fill=False, linewidth=2, color="green"))
        plt.title(methods_name[j])
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
plt.show()


