import numpy as np
import cv2
import matplotlib.pyplot as plt

image = np.zeros((3, 4, 3), dtype=np.uint8)

image[0, 0] = [255, 0, 0]  # Blue
image[1, 1] = [0, 255, 0]  # Green
image[2, 2] = [0, 0, 255]  # Red
image[0, 3] = [0, 255, 255] # Yellow

large = cv2.resize(
    image,
    (400, 300),
    interpolation=cv2.INTER_NEAREST,
)

cv2.imwrite("outputs/day01_pixels.png", large)

rgb = cv2.cvtColor(large, cv2.COLOR_BGR2RGB)

plt.subplot(1, 2, 1)
plt.imshow(large)
plt.title("Converted to RGB")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rgb)
plt.title("Converted to RGB")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/day01_bgr_rgb.png")

print("shape:", image.shape)
print("dtype:", image.dtype)
print("第2行第3列:",image[1, 2])  # Accessing the pixel at row 2, column 3
print("蓝色通道:")
print(image[:, :, 0])  # Accessing the blue channel