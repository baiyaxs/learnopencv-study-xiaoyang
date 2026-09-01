from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")

#噪声参数
sigma = 20
amount = 0.03
kernel_size = 5

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#随机种子添加噪声
rng = np.random.default_rng(42)

#高斯噪声
gaussian_noise = rng.normal(
    loc=0,
    scale=sigma,
    size=gray.shape,
)

gaussian_noisy = (
    gray.astype(np.float32) + gaussian_noise
)

gaussian_noisy = np.clip(
    gaussian_noisy,
    0,
    255,
).astype(np.uint8)

#椒盐噪声
salt_pepper_noisy = gray.copy()
random_map = rng.random(gray.shape)

salt_pepper_noisy[random_map < amount / 2] = 0

salt_pepper_noisy[
    (random_map >= amount / 2)
    & (random_map < amount)
] = 255

def apply_filters(noisy, kernel_size):
    mean_result = cv2.blur(
        noisy,
        (kernel_size, kernel_size),
    )

    gaussian_result = cv2.GaussianBlur(
        noisy,
        (kernel_size, kernel_size),
        sigmaX=0,
    )

    median_result = cv2.medianBlur(
        noisy,
        kernel_size,
    )

    return{
        "Mean": mean_result,
        "Gaussian": gaussian_result,
        "Median": median_result,
    }

gaussian_results = apply_filters(gaussian_noisy, kernel_size)
salt_pepper_results = apply_filters(salt_pepper_noisy, kernel_size)

def mse(clean, noisy):
    difference = (
        clean.astype(np.float32)
        - noisy.astype(np.float32)
    )
    return np.mean(difference ** 2)

print("高斯噪声 MSE:", mse(gray, gaussian_noisy))
print("椒盐噪声 MSE:", mse(gray, salt_pepper_noisy))

for current_kernel in [3, 5, 9]:
    gaussian_test = apply_filters(
        gaussian_noisy,
        current_kernel,
    )

    salt_pepper_test = apply_filters(
        salt_pepper_noisy,
        current_kernel,
    )

    print(f"\n核大小: {current_kernel}x{current_kernel}")

    print("高斯噪声: ")
    for name, result in gaussian_test.items():
        print(name, mse(gray, result))

    print("椒盐噪声: ")
    for name, result in salt_pepper_test.items():
        print(name, mse(gray, result))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [
    (gray, "Clean"),
    (gaussian_noisy, f"Gaussian sigma={sigma}"),
    (salt_pepper_noisy, f"Salt & Pepper {amount:.0%}"),
]

for axis, (current, title) in zip(axes, images):
    axis.imshow(current, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

fig.tight_layout()
fig.savefig(root / "outputs" / "day02_noise.png")
plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(16,7))

rows = [
    ("Gaussian Noise", gaussian_noisy, gaussian_test),
    ("Salt & pepper", salt_pepper_noisy, salt_pepper_test),
]

for row_index, (noise_name, noisy, results) in enumerate(rows):
    row_images = [
        ("Clean", gray),
        ("Noisy", noisy),
        ("Mean", results["Mean"]),
        ("Gaussian", results["Gaussian"]),
        ("Median", results["Median"]),
    ]

    for column_index, (filter_name, current) in enumerate(row_images):
        axis = axes[row_index, column_index]
        axis.imshow(current, cmap="gray", vmin=0, vmax=255)
        axis.axis("off")

        error = mse(gray, current)

        if column_index == 0:
            axis.set_title(f"{noise_name}\n{name}")
        else:
            axis.set_title(f"{filter_name}\nMSE={error:.2f}")

fig.tight_layout()
fig.savefig(
    root / "outputs" / "day02_noise_filters.png",
    dpi=150,
)
plt.close(fig)