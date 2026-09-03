from pathlib import Path

import cv2
import numpy as np

#预处理
def find_document_corners(input_image):
    gray = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        sigmaX=0,
    )

    edges = cv2.Canny(
        blurred,
        30,
        90,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    print("轮廓数量:", len(sorted_contours))

    image_height, image_width = input_image.shape[:2]
    image_area = image_height * image_width
    minimum_area = image_area * 0.1

    for contour in sorted_contours:
        area = cv2.contourArea(contour)

        if area < minimum_area:
            break

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return approximation.reshape(4, 2).astype(np.float32)

    raise ValueError("没有找到符合条件的文档四边形")


def order_points(input_points):
    ordered = np.zeros(
        (4, 2),
        dtype = np.float32,
    )
    sums = input_points.sum(axis=1)
    differences = input_points[:, 1] - input_points[:, 0]
    ordered[0] = input_points[np.argmin(sums)]
    ordered[1] = input_points[np.argmin(differences)]
    ordered[2] = input_points[np.argmax(sums)]
    ordered[3] = input_points[np.argmax(differences)]
    return ordered

def warp_document(input_image, input_points):
    ordered_points = order_points(input_points)

    top_left, top_right, bottom_right, bottom_left = ordered_points

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    target_width = int(round(max(width_top, width_bottom)))
    target_height = int(round(target_width * 297 / 210))

    print("自动计算的目标尺寸:", target_width, target_height)

    destination_points = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(
        ordered_points,
        destination_points,
    )

    warped = cv2.warpPerspective(
        input_image,
        transform_matrix,
        (target_width, target_height),
    )

    return warped
        
root = Path(__file__).resolve().parents[1]
image = cv2.imread(str(root / "inputs" / "document.jpg"))

if image is None:
    raise FileNotFoundError("无法读取图片")


print("原图 shape:", image.shape)

document_points = find_document_corners(image)

print("自动检测到的文档角点:")
print(document_points)

ordered_points = order_points(document_points)

print("排序后的文档角点:")
print(ordered_points)

scanned_document = warp_document(
    image,
    document_points,
)

print("扫描结果 shape:", scanned_document.shape)

output_path = root / "outputs" / "day04_scanner_result.png"
cv2.imwrite(str(output_path), scanned_document)

print("扫描结果保存到:", output_path)

scanned_gray = cv2.cvtColor(
    scanned_document,
    cv2.COLOR_BGR2GRAY,
)

scanned_binary = cv2.adaptiveThreshold(
    scanned_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,
    10,
)

binary_output_path = root / "outputs" / "day04_scanner_binary.png"
cv2.imwrite(
    str(binary_output_path),
    scanned_binary,
)

print("黑白扫描件 shape:", scanned_binary.shape)
print("最小值:", scanned_binary.min())
print("最大值:", scanned_binary.max())