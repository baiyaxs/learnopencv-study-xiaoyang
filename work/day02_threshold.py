from pathlib import Path

import cv2
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

fixed_results = []

for threshold_value in [100, 150, 200]:
    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
)
    fixed_results.append(
        (f"Fixed T={threshold_value}", binary))

otsu_value, otsu = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

print("Otsu 选择的阈值:", otsu_value)
'''
adaptive = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    10,
)
'''
results = [
    ("Gray", gray),
    *fixed_results,
    (f"Otsu T={otsu_value:.0f}", otsu),
    #("Adaptive", adaptive),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for axis, (title, current) in zip(axes, results):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

fig.tight_layout()
fig.savefig(
    root / "outputs" / "day02_threshold.png",
    dpi=150,
)
plt.close(fig)