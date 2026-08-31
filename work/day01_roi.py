from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError(f"无法读取图片：{root / 'inputs' / 'document.jpg'}")

height, width = image.shape[:2]

x1 = int(width * 0.2)
x2 = int(width * 0.8)
y1 = int(height * 0.2) 
y2 = int(height * 0.8)

crop = image[y1:y2, x1:x2]

marked = image.copy()
cv2.rectangle(
    marked,
    (x1,y1),
    (x2,y2),
    (0, 0, 225),
    5,
)

print("原图 shape:", image.shape)
print("裁剪坐标:", (x1, y1, x2, y2))
print("裁剪图 shape:", crop.shape)

cv2.imwrite(str(root / "outputs" / "day01_crop.png"), crop)
cv2.imwrite(str(root / "outputs" / "day01_marked.png"), marked)

matrix = np.zeros((4, 4), dtype=np.uint8)

roi_view = matrix[1:3, 1:3]
roi_view[:] = 255

print("修改切片后的原矩阵: ")
print(matrix)

matrix = np.zeros((4, 4), dtype=np.uint8)

roi_copy = matrix[1:3, 1:3].copy()
roi_copy[:] = 255

print("修改 copy 后的原矩阵: ")
print(matrix)

print("单独的 ROI copy: ")
print(roi_copy)