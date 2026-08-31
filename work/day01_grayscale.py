from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

#导入图片
root = Path(__file__).resolve().parents[1]
input_path = root / "inputs" / "document.jpg"
output_path = root / "outputs" / "day01_gray.png"

image = cv2.imread(str(input_path))

if image is None:
    raise FileNotFoundError(f"无法读取图片：{input_path}")

#计算图片的灰度图
print("shape:", image.shape)
print("dtype:", image.dtype)
print("最小值:", image.min())
print("最大值:", image.max())

blue, green, red =cv2.split(image)

#计算灰度图的两种方法

#加权法（手工公式）
manual_gray = 0.114 * blue + 0.587 * green + 0.299 * red
manual_gray = np.rint(manual_gray)
manual_gray = np.clip(manual_gray, 0, 255).astype(np.uint8)

#加权法（OpenCV内置函数）
opencv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#比较两种方法的差异
difference = cv2.absdiff(manual_gray, opencv_gray)
values, counts = np.unique(difference, return_counts=True)

#平均法
average_gray = np.mean(
    image.astype(np.float32),
    axis=2,
)

average_gray = np.rint(average_gray)
average_gray = np.clip(average_gray, 0, 255).astype(np.uint8)

average_difference = cv2.absdiff(
    average_gray,
    opencv_gray,
)

#生成对比图
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(opencv_gray, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("Weighted (OpenCV)")

axes[2].imshow(average_gray, cmap='gray', vmin=0, vmax=255)
axes[2].set_title("Average")

for axis in axes:
    axis.axis("off")

fig.tight_layout()
fig.savefig(root / "outputs" / "day01_gray_comparison.png")
plt.close(fig)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)

print("H 范围:", hue.min(), hue.max())
print("S 范围:", saturation.min(), saturation.max())
print("V 范围:", value.min(), value.max())

fig, axes = plt.subplots(1, 3, figsize=(12,5))

axes[0].imshow(hue, cmap="gray")
axes[0].set_title("Hue")

axes[1].imshow(saturation, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Saturation")

axes[2].imshow(value, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Value")

for axis in axes:
    axis.axis("off")

fig.tight_layout()
fig.savefig(root / "outputs" / "day01_hsv_comparison.png")
plt.close(fig)

#输出各项数值
print("灰度图 shape:", opencv_gray.shape)
print("平均差异:", difference.mean())
print("最大差异:", difference.max())
print("差异分布:", dict(zip(values.tolist(), counts.tolist())))
print("平均法与加权法的平均差异:", average_difference.mean())
print("平均法与加权法的最大差异:", average_difference.max())

cv2.imwrite(str(output_path), opencv_gray)