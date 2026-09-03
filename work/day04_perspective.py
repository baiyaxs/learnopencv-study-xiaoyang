from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")

points = np.array(
    [
        [968, 148],
        [392, 173],
        [335, 955],
        [1159, 894],
    ],
    dtype=np.float32,
)


def order_points(input_points):
    ordered=np.zeros(
    (4, 2),
    dtype=np.float32,
    )
    sums = input_points.sum(axis=1)
    differences = input_points[:, 1] - input_points[:, 0]
    ordered[0] = input_points[np.argmin(sums)]
    ordered[1] = input_points[np.argmin(differences)]
    ordered[2] = input_points[np.argmax(sums)]
    ordered[3] = input_points[np.argmax(differences)]
    return ordered

ordered_points = order_points(points)
print(ordered_points)

'''
shuffled_points = points[[2, 0, 3, 1]]

print("打乱后的输入:")
print(shuffled_points)

print("重新排序后的输出:")
print(order_points(shuffled_points))
'''

top_left, top_right, bottom_right, bottom_left = ordered_points
width_top = np.linalg.norm(top_left - top_right)
width_bottom = np.linalg.norm(bottom_left - bottom_right)
height_left = np.linalg.norm(top_left - bottom_left)
height_right = np.linalg.norm(top_right - bottom_right)
target_width = int(round(max(width_top, width_bottom)))
#target_height = int(round(max(height_left, height_right)))
target_height = int(round(target_width * 297 / 210))

print("上边宽度:", width_top)
print("下边宽度:", width_bottom)
print("左边高度:", height_left)
print("右边高度:", height_right)
print("目标尺寸:", target_width, target_height)

destination_points = np.array(
    [
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1],
    ],
    dtype=np.float32,
)

print("源点:")
print(ordered_points)
print("目标点:")
print(destination_points)

transform_matrix = cv2.getPerspectiveTransform(
    ordered_points,
    destination_points,
)

print("变换矩阵 shape:", transform_matrix.shape)
print("变换矩阵:")
print(transform_matrix)

warped = cv2.warpPerspective(
    image,
    transform_matrix,
    (target_width, target_height)
)

print("展开图 shape:", warped.shape)

output_path = root / "outputs" / "day04_wraped.png"
cv2.imwrite(str(output_path), warped)
print("保存到:", output_path)

#旋转实验
rotated_points = ordered_points[[1, 2, 3, 0]]

rotated_destination_points = np.array(
    [
        [0, 0],
        [target_height - 1, 0],
        [target_height - 1, target_width - 1],
        [0, target_width - 1],
    ],
    dtype=np.float32,
)

rotated_matrix = cv2.getPerspectiveTransform(
    rotated_points,
    rotated_destination_points,
)

rotated_result = cv2.warpPerspective(
    image,
    rotated_matrix,
    (target_height, target_width),
)

cv2.imwrite(
    str(root / "outputs" / "day04_wrong_rotation.png"),
    rotated_result,
)

#镜像实验
mirrored_points = ordered_points[[1, 0, 3, 2]]

mirrored_matrix = cv2.getPerspectiveTransform(
    mirrored_points,
    destination_points,
)

mirrored_result = cv2.warpPerspective(
    image,
    mirrored_matrix,
    (target_width, target_height),
)

cv2.imwrite(
    str(root / "outputs" / "day04_wrong_mirror.png"),
    mirrored_result,
)

#扭曲实验

crossed_points = ordered_points[[0, 2, 1, 3]]

crossed_matrix = cv2.getPerspectiveTransform(
    crossed_points,
    destination_points,
)

crossed_result = cv2.warpPerspective(
    image,
    crossed_matrix,
    (target_width, target_height),
)

cv2.imwrite(
    str(root / "outputs" / "day04_wrong_crossed.png"),
    crossed_result,
)