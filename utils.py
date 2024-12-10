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

def image_show_cv2(img, color_order=cv2.COLOR_BayerBG2BGR):
    # img = cv2.cvtColor(img, color_order)
    cv2.imshow("Raw Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_show_cv2_thumnail(img, color_order=cv2.COLOR_BayerBG2BGR):
    # img = cv2.cvtColor(img, color_order)
    # 缩小图片尺寸 DownScale by 8
    img = cv2.resize(img, (0, 0), fx=0.125, fy=0.125)
    cv2.imshow("Raw Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_show_plt(img, width, height):
    plt.figure(num="Raw Image", figsize=(width / 100, height / 100))
    plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.axis("off")
    # 保存图片
    # plt.savefig("data/raw_image.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
    plt.show()


def image_show_plt_thumnail(img, width, height):
    # 缩小图片尺寸 DownScale by 8
    plt.figure(num="Raw Image", figsize=(width / 800, height / 800))
    plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.axis("off")
    # 保存图片
    # plt.savefig("data/raw_image_thumnail.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
    plt.show()

def image_show_plt_3D(img, width, height):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)
    Z = img
    ax.plot_surface(X, Y, Z, cmap='rainbow', rstride=1, cstride=1)
    plt.show()

def image_show_plt_fake_color(img, width, height, partten):
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
    plt.show()


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

if __name__ == "__main__":
    # test_read_raw_mipi10()
    # test_image_show()
    open_raw_image("data/4608_3456.raw", 4608, 3456, 10)
