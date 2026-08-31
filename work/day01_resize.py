from pathlib import Path
import cv2

root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError(f"无法读取图片：{root / 'inputs' / 'document.jpg'}")

height, width = image.shape[:2]

target_width = 600
resize_ratio = target_width / width
target_height = round(height * resize_ratio)

small = cv2.resize(
    image,
    (target_width, target_height),
    interpolation=cv2.INTER_AREA,
)

print("原图 shape:", image.shape)
print("小图 shape:", small.shape)
print("缩放比例:", resize_ratio)

point_small = (150, 200)

back_scale_x = width / small.shape[1]
back_scale_y = height / small.shape[0]

point_original = (
    round(point_small[0] * back_scale_x), 
    round(point_small[1] * back_scale_y),
)

print("小图坐标:", point_small)
print("原图坐标:", point_original)

small_marked = small.copy()
original_marked = image.copy()

cv2.circle(small_marked, point_small, 8, (0, 0, 255), -1)
cv2.circle(original_marked, point_original, 14, (0, 0, 255), -1)

cv2.imwrite(
    str(root / "outputs" / "day01_resize_small.png"),
    small_marked,
)

cv2.imwrite(
    str(root / "outputs" / "day01_resize_original.png"),
    original_marked,
)