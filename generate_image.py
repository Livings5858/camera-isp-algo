import numpy as np

def generate_yuv420sp_image(width, height, filename):
    # 创建 Y 分量 (亮度)
    y = np.zeros((height, width), dtype=np.uint8)
    
    # 创建 UV 分量 (色度)，U 和 V 分量大小为 (height // 2, width // 2)
    u = np.zeros((height // 2, width // 2), dtype=np.uint8)
    v = np.zeros((height // 2, width // 2), dtype=np.uint8)

    # 填充 Y 分量为渐变值（仅示例，实际值应根据需求设置）
    y[:height//2, :width//2] = 255  # 上半部分为白色
    y[height//2:, :width//2] = 128  # 下半部分为灰色
    y[:height//2, width//2:] = 64   # 右半部分为中灰色
    y[height//2:, width//2:] = 0    # 下半部分为黑色

    # 填充 U 和 V 分量为固定值（可以根据需求变化）
    u[:, :] = 128  # U 值固定为中性
    v[:, :] = 128  # V 值固定为中性

    # 交替存储 U 和 V 分量，UV 数据按顺序存储 U、V，分别占据 (height // 2 * width // 2)
    uv = np.zeros((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u  # 将 U 分量存储在偶数列
    uv[:, 1::2] = v  # 将 V 分量存储在奇数列

    # 合并 Y 分量和 UV 分量，形成 YUV420SP 格式
    yuv_data = np.concatenate((y.flatten(), uv.flatten()))

    # 将 YUV 数据写入文件
    with open(filename, 'wb') as f:
        f.write(yuv_data)


def generate_raw_mipi_10bit_image(width, height, filename):
    # 生成一个 10-bit 灰度图像数据，像素值范围是 0 - 1023
    image_data = np.random.randint(0, 1024, size=(height, width), dtype=np.uint16)

    # 打包过程：每 4 个像素打包成 5 个字节
    packed_data = []

    # 遍历图像数据并打包
    for row in range(height):
        for col in range(0, width, 4):
            # 取 4 个像素
            pixels = image_data[row, col:col+4]

            # 提取每个像素的前 8 位和后 2 位
            packed_bytes = []
            for pixel in pixels:
                # 获取前 8 位
                packed_bytes.append(pixel >> 2)  # 10-bit -> 8-bit
            # 将后 2 位合并成一个字节
            last_byte = 0
            for i, pixel in enumerate(pixels):
                last_byte |= ((pixel & 0x03) << (i * 2))  # 提取每个像素的后 2 位
            packed_bytes.append(last_byte)

            # 将每 4 个像素打包成 5 字节
            packed_data.extend(packed_bytes)

    # 将打包后的数据保存为 RAW 格式
    with open(filename, 'wb') as f:
        f.write(bytearray(packed_data))

def generate_raw_mipi_10bit_gradient(width, height, filename):
    # 生成一个水平渐变图像，每个像素的值范围是 0 - 1023（10-bit）
    gradient = np.linspace(0, 1023, width, dtype=np.uint16)  # 从0到1023的渐变
    gradient_image = np.tile(gradient, (height, 1))  # 创建一个宽度为width，高度为height的渐变图像

    # 打包过程：每 4 个像素打包成 5 个字节
    packed_data = []

    # 遍历图像数据并打包
    for row in range(height):
        for col in range(0, width, 4):
            # 取 4 个像素
            pixels = gradient_image[row, col:col+4]

            # 提取每个像素的前 8 位和后 2 位
            packed_bytes = []
            for pixel in pixels:
                # 获取前 8 位
                packed_bytes.append(pixel >> 2)  # 10-bit -> 8-bit
            # 将后 2 位合并成一个字节
            last_byte = 0
            for i, pixel in enumerate(pixels):
                last_byte |= ((pixel & 0x03) << (i * 2))  # 提取每个像素的后 2 位
            packed_bytes.append(last_byte)

            # 将每 4 个像素打包成 5 字节
            packed_data.extend(packed_bytes)

    # 将打包后的数据保存为 RAW 格式
    with open(filename, 'wb') as f:
        f.write(bytearray(packed_data))

if __name__ == '__main__':
    # 设置图像宽度和高度
    width = 640
    height = 480
    # 生成并保存 RAW MIPI 10-bit 图像
    generate_raw_mipi_10bit_image(width, height, "data/test_raw_mipi_10bit_640x480.raw")
    # 生成并保存 RAW MIPI 10-bit 渐变图像
    generate_raw_mipi_10bit_gradient(width, height, "data/test_raw_mipi_10bit_gradient_640x480.raw")
    # 生成 YUV420SP 测试图像并保存
    generate_yuv420sp_image(width, height, "data/test_image_640x480.yuv")
