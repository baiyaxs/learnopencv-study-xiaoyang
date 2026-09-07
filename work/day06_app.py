import streamlit as st
import cv2
import numpy as np
from day05_robust_scanner import scan_document

@st.cache_data(
        show_spinner=False
)
def cached_scan(
    input_image,
    canny_low,
    canny_high,
):
    return scan_document(
        input_image,
        canny_low=canny_low,
        canny_high=canny_high,
    )

def encode_png(input_image):
    success, encoded_image = cv2.imencode(
        ".png",
        input_image,
    )

    if not success:
        raise ValueError("无法将扫描结果编码为 PNG")

    return encoded_image.tobytes()

#标签栏标题
st.set_page_config(
    page_title="文档扫描器",
    page_icon="📄",
    layout="wide",
)

#网页大标题
st.title("📄 OpenCV 文档扫描器")

#网页文本提示
st.write(
    "上传一张文档照片, 程序将自动检测边界、"
    "校正透视并生成扫描结果。"
)

#侧边栏
st.sidebar.header("扫描函数")

canny_low, canny_high = st.sidebar.slider(
    "Canny 双阈值",
    min_value=0,
    max_value=255,
    value=(30, 90),
    step=5,
)

#上传控件
uploaded_file = st.file_uploader(
    "选择一张文档照片",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("请先上传一张图片")
    st.stop()

#处理上传上去的图片
uploaded_bytes = uploaded_file.getvalue()

encoded_array = np.frombuffer(
    uploaded_bytes,
    dtype=np.uint8,
)

image = cv2.imdecode(
    encoded_array,
    cv2.IMREAD_COLOR,
)

if image is None:
    st.error("无法解码上传的图片")
    st.stop()

#打印shape
st.write("编码数组 shape:", encoded_array.shape)
st.write("OpenCV 图像 shape:", image.shape)

#转RGB顺序
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB,
)

#输出原图并标注
st.image(
    image_rgb,
    caption="上传的原图",
)

st.subheader("扫描状态")

with st.spinner("正在检测文档并生成扫描结果..."):
    result = cached_scan(
        image,
        canny_low,
        canny_high,
    )

show_debug = st.checkbox(
    "显示检测诊断与中间结果"
)

if show_debug:
    st.write("检测诊断:")
    st.json(result["diagnostics"])
    with st.expander("查看文档检测过程"):
        debug_titles = {
            "gray": "1. 灰度图",
            "blurred": "2. 高斯滤波",
            "edges": "3. Canny边缘", 
            "closed_edges": "4. 闭运算边缘",
        }

        debug_columns = st.columns(2)

        for index, (
            debug_name,
            debug_title,
        ) in enumerate(debug_titles.items()):
            with debug_columns[index % 2]:
                st.image(
                    result["debug_images"][
                        debug_name
                    ],
                    caption=debug_title,
                )

if not result["success"]:
    st.error(
        "扫描失败: "
        + result["failure_reason"]
    )
    st.stop()

st.success("文档检测与透视校正成功")

detected_preview = image.copy()

corner_points = (
    result["points"]
    .astype(np.int32)
)

contour_points = corner_points.reshape(
    -1, 
    1,
    2,
)

cv2.polylines(
    detected_preview,
    [contour_points],
    isClosed=True,
    color=(0, 255, 0),
    thickness=5,
)

for corner_index, (x, y) in enumerate(
    corner_points
):
    cv2.circle(
        detected_preview,
        (x, y),
        radius=10,
        color=(0, 0, 255),
        thickness=-1,
    )

    cv2.putText(
        detected_preview,
        str(corner_index + 1),
        (x + 12, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

detected_preview_rgb = cv2.cvtColor(
    detected_preview,
    cv2.COLOR_BGR2RGB,
)

st.image(
    detected_preview_rgb,
    caption="检测到的文档边界与角点",
)

#展示质量
quality = result["quality"]

area_metric, black_metric, white_metric = (
    st.columns(3)
)

with area_metric:
    st.metric(
        "文档面积占比",
        f"{result['area_ratio']:.1%}",
    )

with black_metric:
    st.metric(
        "黑色像素占比",
        f"{quality['black_ratio']:.1%}",
    )

with white_metric:
    st.metric(
        "白色像素占比",
        f"{quality['white_ratio']:.1%}",
    )

for warning in quality["warnings"]:
    st.warning(warning)

#扫描结果对比
st.subheader("扫描结果对比")

#为了方便单独给彩色交换通道开一个函数
scanned_color_rgb = cv2.cvtColor(
    result["color"],
    cv2.COLOR_BGR2RGB,
)

color_column, gray_column, binary_column = (
    st.columns(3)
)

with color_column:
    st.image(
        scanned_color_rgb,
        caption="彩色扫描",
    )
    color_png = encode_png(
        result["color"]
    )
    st.download_button(
        "下载彩色扫描",
        data=color_png,
        file_name="scanned_color.png",
        mime="image/png",
    )

with gray_column:
    st.image(
        result["gray"],
        caption="灰度扫描",
    )
    gray_png = encode_png(
        result["gray"]
    )
    st.download_button(
        "下载灰度扫描",
        data=gray_png,
        file_name="scanned_gray.png",
        mime="image/png",
    )

with binary_column:
    st.image(
        result["binary"],
        caption="黑白扫描",
    )
    binary_png = encode_png(
        result["binary"]
    )
    st.download_button(
        "下载黑白扫描",
        data=binary_png,
        file_name="scanned_binary.png",
        mime="image/png",
    )