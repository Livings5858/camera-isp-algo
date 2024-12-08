import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

def unpack_mipi_raw10(byte_buf):
    data = np.frombuffer(byte_buf, dtype=np.uint8)
    # 5 bytes contain 4 10-bit pixels (5x8 == 4x10)
    b1, b2, b3, b4, b5 = np.reshape(
        data, (data.shape[0]//5, 5)).astype(np.uint16).T
    p1 = (b1 << 2) + ((b5) & 0x3)
    p2 = (b2 << 2) + ((b5 >> 2) & 0x3)
    p3 = (b3 << 2) + ((b5 >> 4) & 0x3)
    p4 = (b4 << 2) + ((b5 >> 6) & 0x3)
    unpacked = np.reshape(np.concatenate(
        (p1[:, None], p2[:, None], p3[:, None], p4[:, None]), axis=1),  4*p1.shape[0])
    return unpacked

def unpack_mipi_raw10_2(byte_buf):
    data = np.frombuffer(byte_buf, dtype=np.uint8)
    # 5 bytes contain 4 10-bit pixels (5x8 == 4x10)
    b1, b2, b3, b4, b5 = np.reshape(
        data, (data.shape[0]//5, 5)).astype(np.uint16).T
    o1 = (b1 << 2) + ((b5) & 0x3)
    o2 = (b2 << 2) + ((b5 >> 2) & 0x3)
    o3 = (b3 << 2) + ((b5 >> 4) & 0x3)
    o4 = (b4 << 2) + ((b5 >> 6) & 0x3)
    unpacked = np.reshape(np.concatenate(
        (o1[:, None], o2[:, None], o3[:, None], o4[:, None]), axis=1),  4*o1.shape[0])
    return unpacked



# 使用np读取RawMIPI10数据
def read_raw_mipi10(path, height, width):
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
    second_byte = np.left_shift(second_byte, 2) + np.right_shift(
        np.bitwise_and(fifth_byte, 12), 2
    )
    third_byte = np.left_shift(third_byte, 2) + np.right_shift(
        np.bitwise_and(fifth_byte, 48), 4
    )
    fourth_byte = np.left_shift(fourth_byte, 2) + np.right_shift(
        np.bitwise_and(fifth_byte, 192), 6
    )

    raw_data = np.zeros(shape=(height, new_width), dtype=np.uint16)
    raw_data[:, 0:new_width:4] = first_byte[:, 0:packed_num_L]
    raw_data[:, 1:new_width:4] = second_byte[:, 0:packed_num_L]
    raw_data[:, 2:new_width:4] = third_byte[:, 0:packed_num_L]
    raw_data[:, 3:new_width:4] = fourth_byte[:, 0:packed_num_L]

    raw_data = raw_data[:, 0:width]
    return raw_data

def test_read_raw_mipi10():
    path = "data/4608_3456.raw"
    height = 3456
    width = 4608
    img = read_raw_mipi10(path, height, width)
    img = img >> 2
    img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    cv2.imwrite("data/test.jpg", img)

def test_unpack_mipi_raw10():
    path = "data/4608_3456.raw"
    height = 3456
    width = 4608
    raw_frame = np.fromfile(path, dtype=np.uint8, count=height*width*5//4)
    img = unpack_mipi_raw10(raw_frame).reshape(height, width, 1)
    img = img >> 2
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    cv2.imwrite("data/test.jpg", img)

if __name__ == "__main__":
    test_read_raw_mipi10()
