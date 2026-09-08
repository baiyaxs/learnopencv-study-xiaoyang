from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

from day05_robust_scanner import (
    scan_document,
    order_points,
)

root = Path(__file__).resolve().parents[1]

test_cases = [
    {
        "name": "基础文档",
        "path": root / "inputs" / "document.jpg",
        "expected_success": True,
    },
    {
        "name": "彩色文档",
        "path": root / "inputs" / "document_color.png",
        "expected_success": True,
    },
    {
        "name": "空白图片",
        "path": (
            root
            / "inputs"
            / "failure_cases"
            / "blank.png"
        ),
        "expected_success": False,
    },
    {
        "name": "圆形图片",
        "path": (
            root
            / "inputs"
            / "failure_cases"
            / "circle.png"
        ),
        "expected_success": False,
    },
    {
        "name": "小型轮廓",
        "path": (
            root
            / "inputs"
            / "failure_cases"
            / "small_rectangle.png"
        ),
        "expected_success": False,
    },
    {
        "name": "低对比度文档",
        "path": (
            root
            / "outputs"
            / "day05_cases"
            / "low_contrast.png"
        ),
        "expected_success": True,
    },
    {
        "name": "严重模糊文档",
        "path": (
            root
            / "outputs"
            / "day05_cases"
            / "heavy_blur.png"
        ),
        "expected_success": False,
    },
    {
        "name": "角点被裁切",
        "path": (
            root
            / "outputs"
            / "day05_cases"
            / "cropped.png"
        ),
        "expected_success": False,
    },
]

for test_case in test_cases:
    image = cv2.imread(
        str(test_case["path"])
    )

    assert image is not None, (
        f"无法读取测试图片: "
        f"{test_case['path']}"
    )

    start_time = perf_counter()

    result = scan_document(image)

    elapsed_ms = (
        perf_counter() - start_time
    ) * 1000

    actual_success = result["success"]
    expected_success = (
        test_case["expected_success"]
    )

    passed = (
        actual_success
        == expected_success
    )

    print()
    print("测试场景:", test_case["name"])
    print(f"处理耗时: {elapsed_ms:.1f} ms")
    print("预期成功:", expected_success)
    print("实际成功:", actual_success)
    print("测试通过:", passed)

    if not actual_success:
        print(
            "扫描失败原因:",
            result["failure_reason"],
        )

    if actual_success:
        assert result["color"] is not None
        assert result["gray"] is not None
        assert result["binary"] is not None
        assert result["quality"] is not None

        assert result["color"].ndim == 3
        assert result["color"].shape[2] == 3
        assert result["gray"].ndim == 2
        assert result["binary"].ndim == 2

        binary_values = np.unique(
            result["binary"]
        )

        assert np.isin(
            binary_values,
            [0, 255],
        ).all()

        print(
            "彩色输出 shape:",
            result["color"].shape,
        )
        print(
            "灰度输出 shape:",
            result["gray"].shape,
        )
        print(
            "黑白输出 shape:",
            result["binary"].shape,
        )
        print(
            "黑色像素占比:",
            f"{result['quality']['black_ratio']:.1%}",
        )
    else:
        assert result["color"] is None
        assert result["gray"] is None
        assert result["binary"] is None
        assert result["quality"] is None

    assert passed, (
        f"{test_case['name']} 的结果不符合预期"
    )

diamond_points = np.array(
    [
        [50, 0],
        [100, 50],
        [50, 100],
        [0, 50],
    ],
    dtype=np.float32,
)

expected_diamond = np.array(
    [
        [50, 0],
        [100, 50],
        [50, 100],
        [0, 50],
    ],
    dtype=np.float32,
)

ordered_diamond = order_points(diamond_points)

np.testing.assert_array_equal(
    ordered_diamond,
    expected_diamond,
)

unique_point_count = len(
    np.unique(
        ordered_diamond,
        axis=0,
    )
)

print()
print("菱形角点排序测试:")
print(ordered_diamond)
print(
    "排序后不同点数:",
    unique_point_count,
)

assert unique_point_count == 4, (
    "角点排序后出现了重复点"
)

print("全部回归测试通过")