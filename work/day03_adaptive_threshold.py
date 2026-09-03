from pathlib import Path

import cv2
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

# Otsu：整张图只使用一个阈值。
otsu_value, otsu = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

# 自适应阈值：每个像素都根据周围 51×51 邻域计算自己的阈值。
block_size = 51
c_value = 10

adaptive_mean = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    block_size,
    c_value,
)

adaptive_gaussian = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    block_size,
    c_value,
)

print("图像 shape:", gray.shape)
print("Otsu 全局阈值:", otsu_value)
print("自适应阈值 blockSize:", block_size)
print("自适应阈值 C:", c_value)

results = [
    ("Gray", gray),
    (f"Otsu T={otsu_value:.0f}", otsu),
    (f"Adaptive Mean\nblock={block_size}, C={c_value}", adaptive_mean),
    (
        f"Adaptive Gaussian\nblock={block_size}, C={c_value}",
        adaptive_gaussian,
    ),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for axis, (title, current) in zip(axes, results):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

fig.tight_layout()
fig.savefig(root / "outputs" / "day03_adaptive_threshold.png", dpi=150)
plt.close(fig)

