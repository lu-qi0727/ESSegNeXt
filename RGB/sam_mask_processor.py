import os
import cv2
import numpy as np
# import os
#
#
# def remove_mask_from_filenames(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 检查文件名中是否包含 '_mask'
#         if '_mask' in filename:
#             # 生成新的文件名，去除 '_mask'
#             new_filename = filename.replace('_mask', '')
#
#             # 获取旧文件的完整路径
#             old_file = os.path.join(folder_path, filename)
#
#             # 获取新文件的完整路径
#             new_file = os.path.join(folder_path, new_filename)
#
#             # 重命名文件
#             os.rename(old_file, new_file)
#
#             print(f"文件重命名: {filename} -> {new_filename}")
#
#
# # 使用示例，替换为你自己的文件夹路径
# folder_path = r'C:\Users\admin\Desktop\lq-liter\LOSS'
# remove_mask_from_filenames(folder_path)
#
# 定义路径
import cv2
import os
import numpy as np

# 定义输入和输出路径
val_image_dir = r'E:\lq\mmsegmentation-main\data\building_datasets\images\validation'
val_mask_dir = r'C:\Users\admin\Desktop\lq-liter\LOSS1'
val_output_dir = r'E:\lq\mmsegmentation-main\data\building_datasets\images_4channel\validation'

train_image_dir = r'E:\lq\mmsegmentation-main\data\building_datasets\images\training'
train_mask_dir = r'C:\Users\admin\Desktop\lq-liter\LOSS'
train_output_dir = r'E:\lq\mmsegmentation-main\data\building_datasets\images_4channel\training'

# 创建输出文件夹
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(train_output_dir, exist_ok=True)

# 处理图像和掩码的拼接与美化
def process_images(image_dir, mask_dir, output_dir):
    for filename in os.listdir(image_dir):
        # 获取图像和掩码路径
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace(".png", ".png"))  # 假设掩码文件以"_mask"结尾

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            continue

        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取掩码文件: {mask_path}")
            continue

        # 使用伪彩色将掩码映射为颜色图
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # 将原图像和掩码合并，掩码作为第四通道
        image_4channel = np.dstack([image, mask])

        # 保存生成的四通道图像
        output_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_image_path, image_4channel)

        # # 显示美化效果：原图和伪彩色掩码叠加
        # overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        # overlay_output_path = os.path.join(output_dir, filename.replace(".png", "_overlay.png"))
        # cv2.imwrite(overlay_output_path, overlay)

# 处理验证集
process_images(val_image_dir, val_mask_dir, val_output_dir)

# 处理训练集
process_images(train_image_dir, train_mask_dir, train_output_dir)
