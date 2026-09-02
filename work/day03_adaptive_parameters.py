from pathlib import Path
import cv2
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]

image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

def adaptive_gaussian(block_size, c_value):
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )

block_results = []

for block_size in [11, 51, 151]:
    result = adaptive_gaussian(block_size, 10)
    title = f"block={block_size}, C=10"
    block_results.append((title, result))

print(len(block_results))

c_results = []

for c_value in [2, 10, 25]:
    result = adaptive_gaussian(51, c_value)
    title = f"block=51, C={c_value}"
    c_results.append((title, result))

print(len(c_results))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for axis, (title, current) in zip(axes[0], block_results):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

for axis, (title, current) in zip(axes[1], c_results):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

fig.suptitle("Adaptive Gaussian Threshold Parameters")
fig.tight_layout()
fig.savefig(root / "outputs" / "day03_adaptive_parameters.png", dpi=150)
plt.close(fig)