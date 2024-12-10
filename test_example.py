from utils import *
import cv2
import numpy as np

def test_unpack_mipi_raw10():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    cv2.imwrite("data/test_unpack_mipi_raw10.jpg", img)

def test_open_raw_image():
    path = "data/mipi10_4608x3456.raw"
    img = open_raw_image(path, 4608, 3456, 10)
    cv2.imwrite("data/test_open_raw_image.jpg", img)

def test_raw_image_plt_show():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    image_show_plt(img, width, height)
    image_show_plt_thumnail(img, width, height)

def test_image_show_cv2():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    image_show_cv2(img)
    image_show_cv2_thumnail(img)

def test_yuv420sp_to_rgb():
    yuv420sp = np.fromfile("data/test_image_640x480.yuv", dtype=np.uint8, count=640 * 480 * 3 // 2)
    width = 640
    height = 480
    rgb = yuv420sp_to_rgb(yuv420sp, width, height)
    cv2.imwrite("data/test_yuv420sp_to_rgb.jpg", rgb)
    image_show_cv2(rgb)


def test_image_show_plt_3d():
    # 创建一个示例图像数据（例如随机生成的高度数据）
    width, height = 100, 100
    img = np.random.random((height, width))  # 示例：一个随机生成的二维数组

    # 调用 3D 绘制函数
    image_show_plt_3D(img, width, height)

    # 时间太长，注释掉
    # path = "data/mipi10_4608x3456.raw"
    # height = 3456
    # width = 4608
    # raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    # img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    # img = img >> 2
    # img = img.astype(np.uint8)
    # image_show_plt_3D(img, width, height)

def test_image_show_plt_fake_color():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    img = img / 1024.0  # 归一化到 0-1 之间
    image_show_plt_fake_color(img, width, height, "RGGB")

def test_bayer_cumuhistogram():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    img = img >> 2
    img = img.astype(np.uint8)
    R_hist, GR_hist, GB_hist, B_hist = bayer_cumuhistogram(img, "RGGB", max_val=255)
    show_bayer_cumuhistogram(R_hist, GR_hist, GB_hist, B_hist, max_val=255)

def test_bayer_histogram():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    img = img >> 2
    img = img.astype(np.uint8)
    R_hist, GR_hist, GB_hist, B_hist = bayer_histogram(img, "RGGB", max_val=255)
    show_bayer_histogram(R_hist, GR_hist, GB_hist, B_hist, max_val=255)

def test_cases():
    # test_unpack_mipi_raw10()
    # test_open_raw_image()
    # test_raw_image_plt_show()
    # test_image_show_cv2()
    # test_yuv420sp_to_rgb()
    # test_image_show_plt_3d()
    # test_image_show_plt_fake_color()
    # test_bayer_cumuhistogram()
    test_bayer_histogram()

if __name__ == "__main__":
    test_cases()
