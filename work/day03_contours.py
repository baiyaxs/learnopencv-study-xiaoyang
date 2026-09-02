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

edges = run_canny(30, 90)

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE,
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

closed_contours, hierarchy = cv2.findContours(
    closed_edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE,
)

print("轮廓数量:", len(closed_contours))

sorted_contours = sorted(
    closed_contours,
    key=cv2.contourArea,
    reverse=True,
)

top_contours = sorted_contours[:10]

for index, contour in enumerate(top_contours, start=1):
    area = cv2.contourArea(contour)
    print(f"第{index}名, 面积: {area}")

contour_view = image.copy()

cv2.drawContours(
    contour_view,
    top_contours,
    -1,
    (0, 255, 0),
    3,
)

cv2.imwrite(
    str(root / "outputs" / "day03_top_contours.png"),
    contour_view,
)

document_contour = top_contours[0]

perimeter = cv2.arcLength(document_contour, True)
epsilon = 0.02 * perimeter

approx = cv2.approxPolyDP(
    document_contour,
    epsilon,
    True,
)

print("原始轮廓点数:", len(document_contour))
print("轮廓周长:", perimeter)
print("epsilon:", epsilon)
print("近似后的顶点数:", len(approx))
print("顶点坐标:")
print(approx.reshape(-1, 2))

image_area = gray.shape[0] * gray.shape[1]
min_document_area = 0.10 * image_area

document_candidate = None
candidate_area = 0.0

for candidate_contour in sorted_contours:
    area = cv2.contourArea(candidate_contour)

    if area < min_document_area:
        break

    perimeter = cv2.arcLength(candidate_contour, True)
    epsilon = 0.02 * perimeter
    candidate_approx = cv2.approxPolyDP(
        candidate_contour,
        epsilon,
        True,
    )

    if(
        len(candidate_approx) == 4
        and cv2.isContourConvex(candidate_approx)
    ):
        document_candidate = candidate_approx
        candidate_area = area
        break

if document_candidate is None:
    raise RuntimeError("没有找到合格的文档四边形")

print("原图面积:", image_area)
print("最小候选面积:", min_document_area)
print("文档候选面积:", candidate_area)
print("文档候选顶点:")
print(document_candidate.reshape(-1, 2))

result_image = image.copy()

cv2.polylines(
    result_image,
    [document_candidate],
    True,
    (0, 0, 255),
    5,
)

for x, y in document_candidate.reshape(-1, 2):
    cv2.circle(
        result_image,
        (int(x), int(y)),
        12,
        (0, 255, 0),
        -1,
    )

cv2.imwrite(
    str(root / "outputs" / "day03_document_candidate.png"),
    result_image,
)