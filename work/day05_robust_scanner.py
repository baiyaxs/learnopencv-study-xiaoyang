from pathlib import Path

import cv2
import numpy as np

#预处理
def find_document_corners(
        input_image,
        canny_low=30,
        canny_high=90,
        minimum_area_ratio=0.1,
        ):
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
        canny_low,
        canny_high,
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

    debug_images = {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "closed_edges": closed_edges,
    }

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
    minimum_area = image_area * minimum_area_ratio

    if sorted_contours:
        largest_contour_area = cv2.contourArea(
            sorted_contours[0]
        )
    else:
        largest_contour_area = 0.0

    largest_contour_ratio = (
        largest_contour_area / image_area
    )

    diagnostics = {
        "contour_count": len(sorted_contours),
        "largest_contour_ratio": largest_contour_ratio,
        "minimum_area_ratio": minimum_area / image_area,
        "checked_candidate_count": 0,
    }

    print(
        f"最大轮廓占比: {largest_contour_ratio:.1%}, "
        f"最低要求: {minimum_area / image_area:.1%}"
    )

    for contour in sorted_contours:
        area = cv2.contourArea(contour)

        if area < minimum_area:
            break
        diagnostics["checked_candidate_count"] += 1

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
            document_points = approximation.reshape(
                4,
                2,
            ).astype(np.float32)
            return document_points, debug_images, diagnostics

    return None, debug_images, diagnostics

def validate_document_points(input_points, image_shape):
    if input_points.shape != (4, 2):
        raise ValueError(
            f"角点形状应为 (4, 2), 实际为 {input_points.shape}"
        )

    if not np.isfinite(input_points).all():
        raise ValueError("角点坐标包含无效数值")

    unique_points = np.unique(
        input_points,
        axis=0,
    )

    if len(unique_points) != 4:
        raise ValueError("四个文档角点必须互不相同")

    image_height, image_width = image_shape[:2]

    image_diagonal = np.hypot(
        image_width,
        image_height,
    )

    minimum_corner_distance = image_diagonal * 0.02

    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            distance = np.linalg.norm(
                input_points[first_index]
                - input_points[second_index]
            )

            if distance < minimum_corner_distance:
                raise ValueError(
                    "存在过近的文档角点"
                )

    convex_hull = cv2.convexHull(
        input_points.astype(np.float32)
    )

    if len(convex_hull) != 4:
        raise ValueError("四个角点无法组成凸四边形")

    document_area = cv2.contourArea(convex_hull)

    image_height, image_width = image_shape[:2]
    image_area = image_height * image_width
    area_ratio = document_area / image_area

    if not 0.1 <= area_ratio <= 0.95:
        raise ValueError(
            f"文档面积占比异常: {area_ratio:.1%}"
        )

    return area_ratio

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

def get_detection_failure_reason(diagnostics):
    contour_count = diagnostics["contour_count"]
    largest_ratio = diagnostics["largest_contour_ratio"]
    minimum_ratio = diagnostics["minimum_area_ratio"]
    checked_count = diagnostics["checked_candidate_count"]

    if contour_count == 0:
        return "没有检测到任何边缘轮廓"

    if largest_ratio < minimum_ratio:
        return (
            "检测到的轮廓面积过小:"
            f"最大 {largest_ratio :.1%},"
            f"最低要求 {minimum_ratio:.1%}"
        )

    return (
        f"检查了 {checked_count} 个面积合格轮廓，"
        "但都不是凸四边形"
    )

def analyze_scan_quality(binary_image):
    if binary_image.ndim != 2:
        raise ValueError("黑白扫描结果必须是单通道图像")

    unique_values = np.unique(binary_image)

    if not np.isin(
        unique_values,
        [0, 255],
    ).all():
        raise ValueError("黑白扫描结果包含非二值像素")

    black_ratio = float(
        np.mean(binary_image == 0)
    )

    white_ratio = 1.0 - black_ratio
    warnings = []

    if black_ratio < 0.001:
        warnings.append("扫描结果几乎全白")

    elif black_ratio > 0.6:
        warnings.append("扫描结果黑色区域过多")

    return {
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "warnings": warnings,
    }

def scan_document(input_image):
    document_points, debug_images, diagnostics = (
        find_document_corners(input_image)
    )

    result = {
        "success": False,
        "failure_reason": None,
        "points": document_points,
        "area_ratio": None,
        "diagnostics": diagnostics,
        "debug_images": debug_images,
        "color": None,
        "gray": None,
        "binary": None,
        "quality": None,
    }

    if document_points is None:
        result["failure_reason"] = (
            get_detection_failure_reason(
                diagnostics
            )
        )
        return result

    try:
        area_ratio = validate_document_points(
            document_points,
            input_image.shape,
        )

    except ValueError as error:
        result["failure_reason"] = str(error)
        return result
    
    scanned_color = warp_document(
        input_image,
        document_points,
    )

    scanned_gray = cv2.cvtColor(
        scanned_color,
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

    quality = analyze_scan_quality(
        scanned_binary
    )

    result.update({
        "success": True,
        "area_ratio": area_ratio,
        "color": scanned_color,
        "gray": scanned_gray,
        "binary": scanned_binary,
        "quality": quality,
    })

    return result

def main():
    root = Path(__file__).resolve().parents[1]
    image = cv2.imread(str(root / "inputs" / "document.jpg"))

    if image is None:
        raise FileNotFoundError("无法读取图片")


    print("原图 shape:", image.shape)

    debug_output_directory = (
        root / "outputs" / "day05_debug"
    )

    debug_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = scan_document(image)

    print("检测诊断:")
    print(result["diagnostics"])

    for debug_name, debug_image in result["debug_images"].items():
        debug_path = (
            debug_output_directory
            / f"{debug_name}.png"
        )

        saved = cv2.imwrite(
            str(debug_path),
                debug_image,
        )

        if not saved:
            raise OSError(
                f"无法保存中间结果: {debug_path}"
            )
        
    if not result["success"]:
        print("[扫描失败]", result["failure_reason"])
        raise SystemExit(1)

    document_points = result["points"]
    area_ratio = result["area_ratio"]
    scanned_document = result["color"]
    scanned_gray = result["gray"]
    scanned_binary = result["binary"]
    quality = result["quality"]

    #角点打印
    print("中间结果名称:", list(result["debug_images"].keys()))
    print("自动检测到的文档角点:")
    print(document_points)

    ordered_points = order_points(document_points)

    print("排序后的文档角点:")
    print(ordered_points)


    #图片保存
    print("扫描结果 shape:", scanned_document.shape)

    output_path = root / "outputs" / "day05_scanner_result.png"
    cv2.imwrite(str(output_path), scanned_document)

    print("扫描结果保存到:", output_path)

    binary_output_path = root / "outputs" / "day05_scanner_binary.png"
    cv2.imwrite(
        str(binary_output_path),
        scanned_binary,
    )

    #质量报告打印
    print(
        f"黑色像素占比: "
        f"{quality['black_ratio']:.1%}"
    )

    print(
        f"白色像素占比: "
        f"{quality['white_ratio']:.1%}"
    )

    for warning in quality["warnings"]:
        print("[质量警告]", warning)

    print("黑白扫描件 shape:", scanned_binary.shape)
    print("最小值:", scanned_binary.min())
    print("最大值:", scanned_binary.max())
    print(f"文档面积占比: {area_ratio:.1%}")

if __name__ == "__main__":
    main()