import cv2
import numpy as np

from day05_robust_scanner import (
    find_document_corners,
    get_detection_failure_reason,
    analyze_scan_quality,
    scan_document,
)

blank_image = np.full(
    (600, 800, 3),
    255,
    dtype=np.uint8,
)

circle_image = blank_image.copy()

cv2.circle(
    circle_image,
    (400, 300),
    250,
    (0, 0, 0),
    12,
)

small_rectangle_image = blank_image.copy()

cv2.rectangle(
    small_rectangle_image,
    (300, 250),
    (500, 250),
    (0, 0, 0),
    12,
)

test_images = {
    "blank": blank_image,
    "small_rectangle": small_rectangle_image,
    "circle": circle_image,
}

quality_test_images = {
    "all_white": np.full(
        (100, 100),
        255,
        dtype=np.uint8,
    ),
    "all_black": np.zeros(
        (100, 100),
        dtype=np.uint8,
    ),
    "invalid_gray": np.full(
        (100, 100),
        128,
        dtype=np.uint8,
    ),
}

for test_name, test_image in quality_test_images.items():
    print()
    print("质量测试:", test_name)

    try:
        quality = analyze_scan_quality(
            test_image
        )

        print(
            f"黑色占比: "
            f"{quality['black_ratio']:.1%}"
        )
        print("警告:", quality["warnings"])

    except ValueError as error:
        print("[输出错误]", error)

for test_name, test_image in test_images.items():
    points, debug_images, diagnostics = (
        find_document_corners(test_image)
    )
    
    print()
    print("测试场景:", test_name)
    print("诊断数据:", diagnostics)
    print("是否找到文档:", points is not None)
    if points is None:
        failure_reason = get_detection_failure_reason(
            diagnostics
        )
        print("失败原因:", failure_reason)

    print()
    print("统一扫描入口测试：")

for test_name, test_image in test_images.items():
    result = scan_document(test_image)

    print()
    print("场景:", test_name)
    print("成功:", result["success"])
    print("失败原因:", result["failure_reason"])  

