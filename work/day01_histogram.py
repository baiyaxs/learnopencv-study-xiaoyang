from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError(f"无法读取图片：{root / 'inputs' / 'document.jpg'}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256],
).flatten()

p1, median, p99=np.percentile(gray, [1, 50, 99])

print("最小值:", gray.min())
print("最大值:", gray.max())
print("平均值:", gray.mean())
print("标准差:", gray.std())
print("1%分位数:", p1)
print("中位数:", median)
print("99%分位数:", p99)
print("稳健动态范围:", p99 - p1)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Grayscale")
axes[0].axis("off")

axes[1].plot(range(256), histogram)
axes[1].set_title("Histogram")
axes[1].set_xlabel("Intensity")
axes[1].set_ylabel("Pixel Count")
axes[1].set_xlim([0, 255])

fig.tight_layout()
fig.savefig(str(root / "outputs" / "day01_histogram.png"))
plt.close(fig)
