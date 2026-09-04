from pathlib import Path

import cv2

from day05_robust_scanner import (
    find_document_corners,
    get_detection_failure_reason,
)


root = Path(__file__).resolve().parents[1]

original = cv2.imread(
    str(root / "inputs" / "document.jpg")
)

if original is None:
    raise FileNotFoundError("无法读取测试图片")

low_contrast = cv2.convertScaleAbs(
    original,
    alpha=0.25,
    beta=170,
)

heavy_blur = cv2.GaussianBlur(
    original,
    (41, 41),
    sigmaX=0,
)

cropped = original[:, 500:]

test_images = {
    "low_contrast": low_contrast,
    "heavy_blur": heavy_blur,
    "cropped": cropped,
}

output_directory = (
    root / "outputs" / "day05_cases"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

for test_name, test_image in test_images.items():
    cv2.imwrite(
        str(output_directory / f"{test_name}.png"),
        test_image,
    )

    points, debug_images, diagnostics = (
        find_document_corners(test_image)
    )

    print()
    print("测试场景:", test_name)
    print("诊断数据:", diagnostics)

    if points is None:
        print(
            "检测结果: 失败，",
            get_detection_failure_reason(
                diagnostics
            ),
        )
    else:
        print("检测结果: 成功")
        print("角点:")
        print(points)

threshold_pairs = [
    (30, 90),
    (10, 30),
    (5, 15),
]

for low_threshold, high_threshold in threshold_pairs:
    points, _, diagnostics = find_document_corners(
        heavy_blur,
        canny_low=low_threshold,
        canny_high=high_threshold,
    )

    print()
    print(
        "模糊图 Canny:",
        low_threshold,
        high_threshold,
    )
    print("诊断:", diagnostics)
    print("是否成功:", points is not None)