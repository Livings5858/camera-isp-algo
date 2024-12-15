from utils import *
import cv2
import numpy as np
from lsc_map import shading_R, shading_GR, shading_GB, shading_B

def test_unpack_mipi_raw10():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    cv2.imwrite("data/out/test_unpack_mipi_raw10.jpg", img)

def test_open_raw_image():
    path = "data/mipi10_4608x3456.raw"
    img = open_raw_image(path, 4608, 3456, 10)
    cv2.imwrite("data/out/test_open_raw_image.jpg", img)

def test_raw_image_plt_show():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    image_show_plt(img, width, height, "data/out/test_raw_image_plt_show.jpg")
    image_show_plt_thumnail(img, width, height, "data/out/test_raw_image_plt_show_thumnail.jpg")

def test_image_show_cv2():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    image_show_cv2(img, save_path="data/out/test_image_show_cv2.jpg")
    image_show_cv2_thumnail(img, save_path="data/out/test_image_show_cv2_thumnail.jpg")

def test_yuv420sp_to_rgb():
    yuv420sp = np.fromfile("data/out/test_image_640x480.yuv", dtype=np.uint8, count=640 * 480 * 3 // 2)
    width = 640
    height = 480
    rgb = yuv420sp_to_rgb(yuv420sp, width, height)
    image_show_cv2(rgb, save_path="data/out/test_yuv420sp_to_rgb_show.jpg")

def test_image_show_plt_3d():
    # 创建一个示例图像数据（例如随机生成的高度数据）
    width, height = 100, 100
    img = np.random.random((height, width))  # 示例：一个随机生成的二维数组

    # 调用 3D 绘制函数
    image_show_plt_3D(img, width, height, "data/out/test_image_show_plt_3d.jpg")

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
    image_show_plt_fake_color(img, width, height, "RGGB", "data/out/test_image_show_plt_fake_color.jpg")

def test_bayer_cumuhistogram():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    img = img >> 2
    img = img.astype(np.uint8)
    R_hist, GR_hist, GB_hist, B_hist = bayer_cumuhistogram(img, "RGGB", max_val=255)
    show_bayer_cumuhistogram(R_hist, GR_hist, GB_hist, B_hist, max_val=255, save_path="data/out/test_bayer_cumuhistogram.jpg")

def test_bayer_histogram():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    img = img >> 2
    img = img.astype(np.uint8)
    R_hist, GR_hist, GB_hist, B_hist = bayer_histogram(img, "RGGB", max_val=255)
    show_bayer_histogram(R_hist, GR_hist, GB_hist, B_hist, max_val=255, save_path="data/out/test_bayer_histogram.jpg")

def test_binning_image():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    img = img / 1024.0  # 归一化到 0-1 之间
    print(img.shape)
    img = binning_image(img, width, height, 4, 4)
    print(img.shape)

def test_image_show_bayer_channel():
    path = "data/mipi10_4608x3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height * width * 5 // 4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)
    R, GR, GB, B = separate_bayer_channels(img, "RGGB")
    image_show_bayer_channels(R, GR, GB, B, "data/out/test_image_show_bayer_channel.jpg")

def test_lsc_ratio_applied(shading_R, shading_GR, shading_GB, shading_B, shading_ratio):
    path = "data/test_lsc.raw"
    height = 3000
    width = 4032
    raw_frame = np.fromfile(path, dtype=np.uint8)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width)

    # 裁到实际原图
    img = img[:,:4000]
    width = 4000

    # 展示原图
    origin_img = img >> 2
    origin_img = img.astype(np.uint8)
    image_show_plt(origin_img, width, height, "data/out/origin_image.jpg")

    lsc_map_size = (17, 13)
    block_size = max((width // 2 + lsc_map_size[0] - 1) // lsc_map_size[0], (height // 2 + lsc_map_size[1] - 1) // lsc_map_size[1])
    print("block_size: ", block_size)
    new_img = apply_shading_to_image_ratio(img, block_size, shading_R, shading_GR, shading_GB, shading_B, shading_ratio, partten="RGGB")
    
    # 展示apply lsc之后的图
    new_img = new_img >> 2
    new_img = new_img.astype(np.uint8)
    image_show_plt(new_img, width, height, "data/out/lsc_applied_image_ratio.jpg")

def test_cases():
    # test_unpack_mipi_raw10()
    # test_open_raw_image()
    # test_raw_image_plt_show()
    # test_image_show_cv2()
    # test_yuv420sp_to_rgb()
    # test_image_show_plt_3d()
    # test_image_show_plt_fake_color()
    # test_bayer_cumuhistogram()
    # test_bayer_histogram()
    # test_binning_image()
    # test_image_show_bayer_channel()
    test_lsc_ratio_applied(shading_R, shading_GR, shading_GB, shading_B, 0.6)

if __name__ == "__main__":
    test_cases()
