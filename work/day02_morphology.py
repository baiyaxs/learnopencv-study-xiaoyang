from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]

binary = np.zeros((220, 320), dtype=np.uint8)

#主体白色矩形
cv2.rectangle(binary, (60, 50), (260, 170), 255, -1)

#矩形内部的黑色小洞
cv2.circle(binary, (120, 100), 4, 0, -1)
cv2.circle(binary, (190, 130), 6, 0, -1)

#矩形外部的白色噪点
cv2.circle(binary, (30, 30), 3, 255, -1)
cv2.circle(binary, (290, 195), 5, 255, -1)
cv2.circle(binary, (35, 185), 2, 255, -1)

kernel = np.ones((5, 5), dtype=np.uint8)

eroded = cv2.erode(
    binary,
    kernel,
    iterations=1,
)

dilated = cv2.dilate(
    binary,
    kernel,
    iterations=1,
)

opened = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1,
)

closed = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel,
)

results = [
    ("Origianl", binary),
    ("Erosion", eroded),
    ("Dilation", dilated),
    ("Opening", opened),
    ("Closing", closed),
]

fig, axes = plt.subplots(1, 5, figsize=(17, 4))

for axis, (title, current) in zip(axes, results):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

fig.tight_layout()
fig.savefig(
    root / "outputs" / "day02_morphology.png",
    dpi=150,
)
plt.close(fig)

image = cv2.imread(str(root / "inputs" / "document.jpg"))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_sizes=[3, 5, 9]
closed_results = []

for size in kernel_sizes:
    current_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (size, size)
    )

    current_closed = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        current_kernel,
        iterations=3,
    )
    
    closed_results.append((current_closed))

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original gray")
axes[0].axis("off")

for axis, size, current in zip(
    axes[1:],
    kernel_sizes,
    closed_results,
):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"Closing {size}x{size}, iter=3")
    axis.axis("off")

fig.tight_layout()
fig.savefig(
    root / "outputs" / "day02_document_closing.png",
    dpi=150,
)
plt.close(fig)