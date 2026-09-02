from pathlib import Path

import cv2
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]

image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

def run_canny(low_threshold, high_threshold):
    return cv2.Canny(blurred, low_threshold, high_threshold)

canny_results = []
for low_threshold, high_threshold in [(30, 90), (50, 150), (100, 200)]:
    result = run_canny(low_threshold, high_threshold)
    title = f"low={low_threshold}, high={high_threshold}"
    canny_results.append((title, result))

print(len(canny_results))

results = [
    ("Gray", gray),
    *canny_results,
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for axis, (title, current) in zip(axes, results):
    axis.imshow(current, cmap="gray" ,vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

fig.tight_layout()
fig.savefig(str(root / "outputs" / "day03_canny.png"), dpi=150)
plt.close(fig)