import os
import cv2
import random
import numpy as np

# 输入和输出文件夹路径
input_folder = "images"
output_folder = "blurred_images"

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有JPG图片
file_list = [filename for filename in os.listdir(input_folder) if filename.endswith(".JPG")]
random.shuffle(file_list)  # 随机打乱图像顺序

# 随机选择一半图像进行模糊处理
blur_indices = random.sample(range(len(file_list)), len(file_list) // 2)

for idx, filename in enumerate(file_list):
    img = cv2.imread(os.path.join(input_folder, filename))

    if idx in blur_indices:
        # 随机应用高斯模糊或运动模糊
        ksize = random.randint(3, 21)
        if ksize % 2 == 0:
            ksize += 1  # 确保ksize是奇数

        # 选择高斯模糊或运动模糊
        if random.random() > 0.5:
            # 高斯模糊
            blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        else:
            # 运动模糊
            motion_blur_size = random.randint(1, 10)
            kernel = np.zeros((motion_blur_size, motion_blur_size))
            kernel[int((motion_blur_size - 1) / 2), :] = np.ones(motion_blur_size)
            kernel = kernel / motion_blur_size
            blurred_img = cv2.filter2D(img, -1, kernel)
    else:
        # 随机区域中值模糊
        h, w = img.shape[:2]
        region_w = w // 5
        region_h = h // 5

        # 随机选取模糊区域的左上角
        x_start = random.randint(0, w - region_w)
        y_start = random.randint(0, h - region_h)

        # 获取要模糊的区域
        region = img[y_start:y_start + region_h, x_start:x_start + region_w]

        # 应用中值模糊
        region_blurred = cv2.medianBlur(region, random.choice([3, 5, 7]))

        # 将模糊区域放回原图
        img[y_start:y_start + region_h, x_start:x_start + region_w] = region_blurred
        blurred_img = img

    # 保存处理后的图片
    cv2.imwrite(os.path.join(output_folder, filename), blurred_img)

print("所有图片处理完成并保存在", output_folder, "文件夹下。")