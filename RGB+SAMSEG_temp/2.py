# import os
#
# # 文件路径
# normalized_dir = r'F:\clip\归一化1'  # 归一化后的图像文件夹
#
# # 遍历归一化后的影像文件
# for file_name in os.listdir(normalized_dir):
#     # 检查文件名是否以 'normalized_' 开头并且是 PNG 文件
#     if file_name.startswith('normalized_') and file_name.endswith('.png'):
#         # 获取去掉 'normalized_' 前缀后的新文件名
#         new_name = file_name.replace('normalized_', '', 1)
#
#         # 构建完整的文件路径
#         old_file_path = os.path.join(normalized_dir, file_name)
#         new_file_path = os.path.join(normalized_dir, new_name)
#
#         # 重命名文件
#         try:
#             os.rename(old_file_path, new_file_path)
#             print(f"文件重命名成功: {file_name} -> {new_name}")
#         except Exception as e:
#             print(f"重命名失败: {file_name}, 错误: {e}")
#
#
#
#
#









import os
from PIL import Image
import numpy as np

# 文件路径
image_dir = r'F:\clip\image'           # 原始RGB图像路径
mask_dir = r'F:\clip\归一化1'           # 掩码路径
output_dir = r'F:\clip\归一化后拼接'    # 输出路径

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历image_dir中的文件
for file_name in os.listdir(image_dir):
    # 检查文件是否为PNG文件
    if file_name.endswith('.png'):
        # 构建原始RGB图像和掩码图像的路径
        rgb_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)

        # 读取RGB图像
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"无法读取RGB图像: {rgb_path}, 错误信息: {e}")
            continue

        # 读取掩码图像（单通道）
        try:
            mask_img = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"无法读取掩码图像: {mask_path}, 错误信息: {e}")
            continue

        # 将RGB图像和掩码图像转换为numpy数组
        rgb_array = np.array(rgb_img)
        mask_array = np.array(mask_img)

        # 扩展掩码图像的维度，使其可以与RGB图像拼接
        mask_array_expanded = np.expand_dims(mask_array, axis=2)

        # 将掩码图像与RGB图像拼接，形成4通道图像
        combined_array = np.concatenate((rgb_array, mask_array_expanded), axis=2)

        # 将拼接后的数组转换为图像对象
        combined_img = Image.fromarray(combined_array)

        # 保存拼接后的图像
        output_path = os.path.join(output_dir, file_name)
        combined_img.save(output_path)
        print(f"成功保存拼接图像: {output_path}")
