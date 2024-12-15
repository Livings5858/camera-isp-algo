import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

# 使用np读取RawMIPI10数据(弃用)
def _read_raw_mipi10(path, height, width):
    new_width = int(math.floor((width + 3) / 4) * 4)
    packed_num_L = int(new_width / 4)
    width_byte_num = packed_num_L * 5
    width_byte_num = int(math.floor((width_byte_num + 7) / 8) * 8)
    image_byte = height * width_byte_num
    raw_frame = np.fromfile(path, dtype=np.uint8, count=image_byte)
    raw_frame.shape = [height, width_byte_num]

    # astype('uint16')防止下面计算时溢出
    first_byte = raw_frame[:, 0:image_byte:5]
    second_byte = raw_frame[:, 1:image_byte:5]
    third_byte = raw_frame[:, 2:image_byte:5]
    fourth_byte = raw_frame[:, 3:image_byte:5]
    fifth_byte = raw_frame[:, 4:image_byte:5]

    first_byte = first_byte.astype("uint16")
    second_byte = second_byte.astype("uint16")
    third_byte = third_byte.astype("uint16")
    fourth_byte = fourth_byte.astype("uint16")
    fifth_byte = fifth_byte.astype("uint16")

    first_byte = np.left_shift(first_byte, 2) + np.bitwise_and(fifth_byte, 3)
    second_byte = np.left_shift(second_byte, 2) + np.right_shift(np.bitwise_and(fifth_byte, 12), 2)
    third_byte = np.left_shift(third_byte, 2) + np.right_shift(np.bitwise_and(fifth_byte, 48), 4)
    fourth_byte = np.left_shift(fourth_byte, 2) + np.right_shift(np.bitwise_and(fifth_byte, 192), 6)

    raw_data = np.zeros(shape=(height, new_width), dtype=np.uint16)
    raw_data[:, 0:new_width:4] = first_byte[:, 0:packed_num_L]
    raw_data[:, 1:new_width:4] = second_byte[:, 0:packed_num_L]
    raw_data[:, 2:new_width:4] = third_byte[:, 0:packed_num_L]
    raw_data[:, 3:new_width:4] = fourth_byte[:, 0:packed_num_L]

    raw_data = raw_data[:, 0:width]
    return raw_data

def image_show_cv2(img, color_order=cv2.COLOR_BayerBG2BGR, save_path=None):
    # img = cv2.cvtColor(img, color_order)
    if save_path is not None:
        cv2.imwrite(save_path, img)
    else:
        cv2.imshow("Raw Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def image_show_cv2_thumnail(img, color_order=cv2.COLOR_BayerBG2BGR, save_path=None):
    # img = cv2.cvtColor(img, color_order)
    # 缩小图片尺寸 DownScale by 8
    img = cv2.resize(img, (0, 0), fx=0.125, fy=0.125)
    if save_path is not None:
        cv2.imwrite(save_path, img)
    else:
        cv2.imshow("Raw Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def image_show_plt(img, width, height, save_path=None):
    plt.figure(num="Raw Image", figsize=(width / 100, height / 100))
    plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

def image_show_plt_thumnail(img, width, height, save_path=None):
    # 缩小图片尺寸 DownScale by 8
    plt.figure(num="Raw Image", figsize=(width / 800, height / 800))
    plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

def image_show_plt_3D(img, width, height, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)
    Z = img
    ax.plot_surface(X, Y, Z, cmap='rainbow', rstride=1, cstride=1)
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

def image_show_plt_fake_color(img, width, height, partten, save_path=None):
    x = width / 100
    y = height / 100
    rgb_img = np.zeros(shape = (height, width, 3))
    R = rgb_img[:, :, 0]
    # GR/GB 使用相同的颜色通道
    GR = rgb_img[:, :, 1]
    GB = rgb_img[:, :, 1]
    B = rgb_img[:, :, 2]

    if partten == "RGGB":
        R[::2, ::2] = img[::2, ::2]
        GR[::2, 1::2] = img[::2, 1::2]
        GB[1::2, ::2] = img[1::2, ::2]
        B[1::2, 1::2] = img[1::2, 1::2]
    elif partten == "BGGR":
        B[::2, ::2] = img[::2, ::2]
        GB[::2, 1::2] = img[::2, 1::2]
        GR[1::2, ::2] = img[1::2, ::2]
        R[1::2, 1::2] = img[1::2, 1::2]
    elif partten == "GRBG":
        GR[::2, ::2] = img[::2, ::2]
        R[::2, 1::2] = img[::2, 1::2]
        B[1::2, ::2] = img[1::2, ::2]
        GB[1::2, 1::2] = img[1::2, 1::2]
    elif partten == "GBRG":
        GB[::2, ::2] = img[::2, ::2]
        B[::2, 1::2] = img[::2, 1::2]
        R[1::2, ::2] = img[1::2, ::2]
        GR[1::2, 1::2] = img[1::2, 1::2]
    else:
        print("unsupport partten:", partten)

    plt.figure(num="Fake color Raw Image", figsize=(x, y))
    plt.imshow(rgb_img, interpolation="bicubic", vmax=1.0, vmin=0.0)
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

def image_show_bayer_channels(R, Gr, Gb, B, save_path=None):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(R, cmap='gray')
    axs[0].set_title("Red Channel")
    axs[1].imshow(Gr, cmap='gray')
    axs[1].set_title("Green (Red Row) Channel")
    axs[2].imshow(Gb, cmap='gray')
    axs[2].set_title("Green (Blue Row) Channel")
    axs[3].imshow(B, cmap='gray')
    axs[3].set_title("Blue Channel")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

# 分离Bayer raw通道
def separate_bayer_channels(img, bayer_pattern):
    if bayer_pattern == "RGGB":
        R = img[::2, ::2]
        GR = img[::2, 1::2]
        GB = img[1::2, ::2]
        B = img[1::2, 1::2]
    elif bayer_pattern == "BGGR":
        B = img[::2, ::2]
        GB = img[::2, 1::2]
        GR = img[1::2, ::2]
        R = img[1::2, 1::2]
    elif bayer_pattern == "GRBG":
        GR = img[::2, ::2]
        R = img[::2, 1::2]
        B = img[1::2, ::2]
        GB = img[1::2, 1::2]
    elif bayer_pattern == "GBRG":
        GB = img[::2, ::2]
        B = img[::2, 1::2]
        R = img[1::2, ::2]
        GR = img[1::2, 1::2]
    else:
        print("unsupport bayer pattern:", bayer_pattern)

    return R, GR, GB, B

def bayer_channel_intergration(R, GR, GB, B, bayer_pattern):
    img = np.zeros((R.shape[0] * 2, R.shape[1] * 2), dtype=np.uint16)
    if bayer_pattern == "RGGB":
        img[::2, ::2] = R
        img[::2, 1::2] = GR
        img[1::2, ::2] = GB
        img[1::2, 1::2] = B
    elif bayer_pattern == "BGGR":
        img[::2, ::2] = B
        img[::2, 1::2] = GB
        img[1::2, ::2] = GR
        img[1::2, 1::2] = R
    elif bayer_pattern == "GRBG":
        img[::2, ::2] = GR
        img[::2, 1::2] = R
        img[1::2, ::2] = B
        img[1::2, 1::2] = GB
    elif bayer_pattern == "GBRG":
        img[::2, ::2] = GB
        img[::2, 1::2] = B
        img[1::2, ::2] = R
        img[1::2, 1::2] = GR
    else:
        print("unsupport bayer pattern:", bayer_pattern)

    return img

def sample_separate(img):
    C1 = img[::2, ::2]
    C2 = img[::2, 1::2]
    C3 = img[1::2, ::2]
    C4 = img[1::2, 1::2]
    return C1, C2, C3, C4

def mono_cumuhistogram(img, max_val=255):
    hist, bins = np.histogram(img.flatten(), bins=range(0, max_val + 1))
    # 累积直方图
    cumsum = np.cumsum(hist)
    return cumsum

def mono_average(img):
    return np.mean(img)

def bayer_cumuhistogram(img, partten="RGGB", max_val=255):
    R, GR, GB, B = separate_bayer_channels(img, partten)
    R_hist = mono_cumuhistogram(R, max_val)
    GR_hist = mono_cumuhistogram(GR, max_val)
    GB_hist = mono_cumuhistogram(GB, max_val)
    B_hist = mono_cumuhistogram(B, max_val)
    return R_hist, GR_hist, GB_hist, B_hist

def bayer_histogram(img, partten="RGGB", max_val=255):
    R, GR, GB, B = separate_bayer_channels(img, partten)
    R_hist = np.histogram(R, bins=range(0, max_val + 1))
    GR_hist = np.histogram(GR, bins=range(0, max_val + 1))
    GB_hist = np.histogram(GB, bins=range(0, max_val + 1))
    B_hist = np.histogram(B, bins=range(0, max_val + 1))
    return R_hist, GR_hist, GB_hist, B_hist

def bayer_average(img, partten="RGGB"):
    R, GR, GB, B = separate_bayer_channels(img, partten)
    R_avg = mono_average(R)
    GR_avg = mono_average(GR)
    GB_avg = mono_average(GB)
    B_avg = mono_average(B)
    return R_avg, GR_avg, GB_avg, B_avg

def show_bayer_cumuhistogram(R_hist, GR_hist, GB_hist, B_hist, max_val=255, save_path=None):
    # 创建一个2x2的子图网格
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 绘制每个通道的直方图到对应的子图中
    axs[0, 0].bar(np.arange(0, max_val), R_hist, label="Red", color='red', lw=2)
    axs[0, 0].set_title("Red Channel Histogram")
    axs[0, 0].set_xlabel("Pixel Intensity")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].grid(True)
    
    axs[0, 1].bar(np.arange(0, max_val), GR_hist, label="Green (R)", color='green', lw=2)
    axs[0, 1].set_title("Green (R) Channel Histogram")
    axs[0, 1].set_xlabel("Pixel Intensity")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid(True)
    
    axs[1, 0].bar(np.arange(0, max_val), GB_hist, label="Green (B)", color='limegreen', lw=2)
    axs[1, 0].set_title("Green (B) Channel Histogram")
    axs[1, 0].set_xlabel("Pixel Intensity")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(True)
    
    axs[1, 1].bar(np.arange(0, max_val), B_hist, label="Blue", color='blue', lw=2)
    axs[1, 1].set_title("Blue Channel Histogram")
    axs[1, 1].set_xlabel("Pixel Intensity")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图像
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

def show_bayer_histogram(R_hist, GR_hist, GB_hist, B_hist, max_val=255, save_path=None):
    # 创建一个2x2的子图网格
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # 绘制每个通道的直方图到对应的子图中
    axs[0, 0].bar(np.arange(0, max_val), R_hist[0], label="Red", color='red', lw=2)
    axs[0, 0].set_title("Red Channel Histogram")
    axs[0, 0].set_xlabel("Pixel Intensity")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].grid(True)
    
    axs[0, 1].bar(np.arange(0, max_val), GR_hist[0], label="Green (R)", color='green', lw=2)
    axs[0, 1].set_title("Green (R) Channel Histogram")
    axs[0, 1].set_xlabel("Pixel Intensity")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid(True)
    
    axs[1, 0].bar(np.arange(0, max_val), GB_hist[0], label="Green (B)", color='limegreen', lw=2)
    axs[1, 0].set_title("Green (B) Channel Histogram")
    axs[1, 0].set_xlabel("Pixel Intensity")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(True)
    
    axs[1, 1].bar(np.arange(0, max_val), B_hist[0], label="Blue", color='blue', lw=2)
    axs[1, 1].set_title("Blue Channel Histogram")
    axs[1, 1].set_xlabel("Pixel Intensity")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图像
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches = -0.1)
    else:
        plt.show()

def get_region(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def binning_image(img, width, height, bin_size_w, bin_size_h):
    # 计算分块的数量
    bin_num_w = int(width / bin_size_w)
    bin_num_h = int(height / bin_size_h)
    binning_image = np.empty((bin_num_h*2, bin_num_w*2), dtype=np.float32)
    x = 0
    y = 0
    for i in range(bin_num_h):
        for j in range(bin_num_w):
            region_data = get_region(img, x, y, bin_size_w, bin_size_h)
            C1, C2, C3, C4 = sample_separate(region_data)
            binning_image[i*2, j*2] = np.mean(C1)
            binning_image[i*2, j*2+1] = np.mean(C2)
            binning_image[i*2+1, j*2] = np.mean(C3)
            binning_image[i*2+1, j*2+1] = np.mean(C4)
            x += bin_size_w
        y += bin_size_h
        x = 0
    return binning_image

def apply_shading_to_image(img, block_size, shading_R, shading_GR, shading_GB, shading_B, partten="RGGB"):
    R, GR, GB, B = separate_bayer_channels(img, "RGGB")
    HH, HW = R.shape

    # 如果size不整除，需要调整
    size_new_w = (HW + block_size - 1) // block_size * block_size
    size_new_h = (HH + block_size - 1) // block_size * block_size
    size_new = (size_new_w, size_new_h)

    # 插值
    extend_R_gain_map = cv2.resize(shading_R, size_new, interpolation=cv2.INTER_CUBIC)
    extend_GR_gain_map = cv2.resize(shading_GR, size_new, interpolation=cv2.INTER_CUBIC)
    extend_GB_gain_map = cv2.resize(shading_GB, size_new, interpolation=cv2.INTER_CUBIC)
    extend_B_gain_map = cv2.resize(shading_B, size_new, interpolation=cv2.INTER_CUBIC)

    left_padding = (size_new_w - HW) // 2
    top_padding = (size_new_h - HH) // 2

    # 裁剪到原图大小
    R_gain_map = extend_R_gain_map[top_padding : HH + top_padding, left_padding : HW + left_padding]
    GR_gain_map = extend_GR_gain_map[top_padding : HH + top_padding, left_padding : HW + left_padding]
    GB_gain_map = extend_GB_gain_map[top_padding : HH + top_padding, left_padding : HW + left_padding]
    B_gain_map = extend_B_gain_map[top_padding : HH + top_padding, left_padding : HW + left_padding]


    R_new = R * R_gain_map
    GR_new = GR * GR_gain_map
    GB_new = GB * GB_gain_map
    B_new = B * B_gain_map

    display_rggb_3D(R=R, GR=GR, GB=GB, B=B, save_path="data/out/statistics_raw.png")
    display_rggb_3D(R_gain_map, GR_gain_map, GB_gain_map, B_gain_map, save_path="data/out/statistics_gain_maps.png")
    display_rggb_3D(R_new, GR_new, GB_new, B_new, save_path="data/out/statistics_new_raw.png")

    new_img = bayer_channel_intergration(R_new, GR_new, GB_new, B_new, partten)
    new_img = np.clip(new_img, min=0, max=1023)
    return new_img

def apply_shading_to_image_ratio(img, block_size, shading_R, shading_GR, shading_GB, shading_B, shading_ratio, partten="RGGB"):
    luma_shading = (shading_GR + shading_GB) / 2

    R_color_shading = shading_R / luma_shading
    GR_color_shading = shading_GR / luma_shading
    GB_color_shading = shading_GB / luma_shading
    B_color_shading = shading_B / luma_shading

    new_luma_shading = (luma_shading - 1) * shading_ratio + 1

    new_shading_R = R_color_shading * new_luma_shading
    new_shading_GR = GR_color_shading * new_luma_shading
    new_shading_GB = GB_color_shading * new_luma_shading
    new_shading_B = B_color_shading * new_luma_shading

    return apply_shading_to_image(img, block_size, new_shading_R, new_shading_GR, new_shading_GB, new_shading_B, partten)

def display_rggb_3D(R, GR, GB, B, save_path=None, cmap='rainbow'):
    channels = [R, GR, GB, B]
    titles = ['R', 'GR', 'GB', 'B']

    fig = plt.figure(figsize=(40, 10))

    for i, val in enumerate(channels):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        x = np.arange(val.shape[1])
        y = np.arange(val.shape[0])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, val, cmap=cmap, edgecolor='k', linewidth=0.5)
        ax.set_title(titles[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Val')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def pack_mipi_raw10(image_data, filename):
    height, width = image_data.shape
    packed_data = []

    for row in range(height):
        for col in range(0, width, 4):
            pixels = image_data[row, col:col+4]
            packed_bytes = []
            for pixel in pixels:
                packed_bytes.append(pixel >> 2)
            last_byte = 0
            for i, pixel in enumerate(pixels):
                last_byte |= ((pixel & 0x03) << (i * 2))
            packed_bytes.append(last_byte)
            packed_data.extend(packed_bytes)

    with open(filename, 'wb') as f:
        f.write(bytearray(packed_data))

def unpack_mipi_raw10(byte_buf):
    data = np.frombuffer(byte_buf, dtype=np.uint8)
    # 5 bytes contain 4 10-bit pixels (5x8 == 4x10)
    b1, b2, b3, b4, b5 = np.reshape(data, (data.shape[0] // 5, 5)).astype(np.uint16).T
    o1 = (b1 << 2) + ((b5) & 0x3)
    o2 = (b2 << 2) + ((b5 >> 2) & 0x3)
    o3 = (b3 << 2) + ((b5 >> 4) & 0x3)
    o4 = (b4 << 2) + ((b5 >> 6) & 0x3)
    unpacked = np.reshape(np.concatenate((o1[:, None], o2[:, None], o3[:, None], o4[:, None]), axis=1), 4 * o1.shape[0])
    return unpacked

def unpack_mipi_raw12(byte_buf):
    data = np.frombuffer(byte_buf, dtype=np.uint8)
    # 3 bytes contain 2 12-bit pixels (3x8 == 2x12)
    b1, b2, b3 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    o1 = (b1 << 4) + ((b3) & 0xF)
    o2 = (b2 << 4) + ((b3 >> 4) & 0xF)
    unpacked = np.reshape(np.concatenate((o1[:, None], o2[:, None]), axis=1), 2 * o1.shape[0])
    return unpacked


def unpack_mipi_raw14(byte_buf):
    data = np.frombuffer(byte_buf, dtype=np.uint8)
    # 7 bytes contain 4 14-bit pixels (7x8 == 4x14)
    b1, b2, b3, b4, b5, b6, b7 = np.reshape(data, (data.shape[0] // 7, 7)).astype(np.uint16).T
    o1 = (b1 << 6) + (b5 & 0x3F)
    o2 = (b2 << 6) + (b5 >> 6) + (b6 & 0xF)
    o3 = (b3 << 6) + (b6 >> 4) + (b7 & 0x3)
    o4 = (b4 << 6) + (b7 >> 2)
    unpacked = np.reshape(np.concatenate((o1[:, None], o2[:, None], o3[:, None], o4[:, None]), axis=1), 4 * o1.shape[0])
    return unpacked


def align_down(size, align):
    return size & ~((align) - 1)


def align_up(size, align):
    return align_down(size + align - 1, align)


def remove_padding(data, width, height, bit_deepth):
    buff = data
    real_width = int(width / 8 * bit_deepth)
    align_width = align_up(real_width, 32)
    align_height = align_up(height, 16)

    buff = buff.reshape(align_height, align_width)
    buff = buff[:height, :real_width]
    buff = buff.reshape(height * real_width)
    return buff


def save_image(img, name):
    cv2.imwrite(name, img)


def open_raw_image(mipiFile, imgWidth, imgHeight, bitDeepth, bayerOrder=cv2.COLOR_BayerBG2BGR):
    mipiData = np.fromfile(mipiFile, dtype="uint8")
    mipiData = remove_padding(mipiData, imgWidth, imgHeight, bitDeepth)

    if bitDeepth == 8 or bitDeepth == 16:
        print("raw8 and raw16 no need to unpack")
        img = mipiData.reshape(imgHeight, imgWidth, 1)
        return img
    elif bitDeepth == 10:
        # raw10
        bayerData = unpack_mipi_raw10(mipiData)
        img = bayerData >> 2
    elif bitDeepth == 12:
        # raw12
        bayerData = unpack_mipi_raw12(mipiData)
        img = bayerData >> 4
    elif bitDeepth == 14:
        # raw14
        bayerData = unpack_mipi_raw14(mipiData)
        img = bayerData >> 6
    else:
        print("unsupport bayer bitDeepth:", bitDeepth)

    # bayerData.tofile(mipiFile[:-4]+'_unpack.raw')
    img = img.astype(np.uint8).reshape(imgHeight, imgWidth, 1)
    # # 显示图片
    # image_show_plt_thumnail(img, imgWidth, imgHeight)
    # # 保存图片
    # rgbimg = cv2.cvtColor(img, bayerOrder)
    # cv2.imwrite(mipiFile[:-4]+'_unpack.jpg', rgbimg)
    return img


def yuv420sp_to_rgb(yuv420sp, width, height):
    # YUV420SP 格式：Y 分量 (宽 * 高)，UV 分量 (宽 / 2 * 高 / 2 * 2)
    
    # 1. 解析 YUV 数据
    Y = yuv420sp[:width * height].reshape((height, width))  # Y 分量
    UV = yuv420sp[width * height:].reshape((height // 2, width // 2, 2))  # UV 分量
    
    # 2. 扩展 UV 分量到全图
    U = UV[:,:,0].repeat(2, axis=0).repeat(2, axis=1)  # U 分量
    V = UV[:,:,1].repeat(2, axis=0).repeat(2, axis=1)  # V 分量
    
    # 3. 将 YUV 转换为 RGB
    R = Y + 1.402 * (V - 128)
    G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
    B = Y + 1.772 * (U - 128)
    
    # 4. 限制 RGB 值在 0 到 255 之间
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    
    # 5. 合并 RGB 分量
    rgb_image = np.stack([R, G, B], axis=-1).astype(np.uint8)
    
    return rgb_image

