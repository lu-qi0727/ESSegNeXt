import os
from PIL import Image
import numpy as np


def process_images_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理图像文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)

            # 打开图像
            image = Image.open(image_path)
            # 将图像转换为numpy数组
            pixel_values = np.array(image)

            # 将像素值为255的设置为1
            pixel_values[pixel_values == 255] = 1

            # 将处理后的数组转换回图像
            processed_image = Image.fromarray(pixel_values.astype(np.uint8))

            # 保存覆盖原影像
            processed_image.save(image_path)


# 替换为你的文件夹路径
folder_path = r"E:\lq\mmsegmentation-main\data\building_datasets\annotation\validation"  # 修改为你的文件夹路径
process_images_in_folder(folder_path)
print("处理完成！")
# 定义文件夹路径
# folder_path = r"E:\lq\mmsegmentation-main\data\building_datasets\annotation\training"  # 修改为你的文件夹路径
#
# # 处理文件夹中的所有图片
# process_images_in_folder(folder_path)





# from PIL import Image
# import numpy as np
#
#
# def get_unique_pixel_values(image_path):
#     # 打开图像
#     image = Image.open(image_path)
#
#     # 将图像转换为numpy数组
#     pixel_values = np.array(image)
#
#     # 获取图像的所有像素值
#     unique_values = np.unique(pixel_values)
#
#     # 返回像素值的数量和具体值
#     return len(unique_values), unique_values
#
#
# # 替换为你的图像文件路径
# image_path = r"E:\lq\mmsegmentation-main\data\building_datasets\annotation\training\42.png"
# count, unique_values = get_unique_pixel_values(image_path)
#
# print(f'像素值的数量: {count}')
# print(f'不同的像素值: {unique_values}')
