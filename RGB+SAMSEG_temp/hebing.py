
# import os
# import numpy as np
# from PIL import Image
#
# def merge_images(image_dir, output_path):
#     images = []
#     for filename in os.listdir(image_dir):
#         if filename.endswith('.tif') or filename.endswith('.jpg'):
#             img_path = os.path.join(image_dir, filename)
#             img = Image.open(img_path)
#             images.append(np.array(img))
#
#     if not images:
#         print("No images found in the directory.")
#         return
#
#     num_images = len(images)
#     rows = int(np.ceil(np.sqrt(num_images)))
#     cols = int(np.ceil(num_images / rows))
#
#     first_img = images[0]
#     img_height, img_width, _ = first_img.shape
#     merged_image = np.zeros((rows * img_height, cols * img_width, 3), dtype=np.uint8)
#
#     for idx, img in enumerate(images):
#         row_idx = idx // cols
#         col_idx = idx % cols
#         merged_image[row_idx * img_height:(row_idx + 1) * img_height,
#                      col_idx * img_width:(col_idx + 1) * img_width] = img
#
#     merged_image = Image.fromarray(merged_image)
#     merged_image.save(output_path)
#     print(f"Merged {num_images} images into {output_path}.")
#
# # Example usage
# image_dir = 'D:/lq/mmsegmentation-main/data/building_datasets/img_dir/result3'
# output_path = 'D:/lq/mmsegmentation-main/data/building_datasets/img_dir/hebing3.png'
# merge_images(image_dir, output_path)


#
#
# from osgeo import gdal
# import numpy as np
# import os
#
#
# def stitch_images(input_dir, output_file, original_image_path):
#     # 打开原始影像以获取元数据
#     original_ds = gdal.Open(original_image_path)
#     img_width = original_ds.RasterXSize
#     img_height = original_ds.RasterYSize
#     bands_count = original_ds.RasterCount
#
#     # 创建一个空数组用于存储拼接结果
#     stitched_image = np.zeros((bands_count, img_height, img_width), dtype='uint8')
#
#     # 确定每个裁剪块的大小
#     block_size = 512
#     tile_count = 0
#
#     # 遍历文件夹中的所有影像块
#     for j in range(int(np.ceil(img_height / block_size))):  # 行
#         for i in range(int(np.ceil(img_width / block_size))):  # 列
#             tile_path = os.path.join(input_dir, f'cropped_image_{i}_{j}.tif')
#
#             # 确保文件存在
#             if not os.path.exists(tile_path):
#                 print(f"Warning: Tile {tile_path} does not exist. Skipping this tile.")
#                 continue
#
#             tile_ds = gdal.Open(tile_path)
#             tile_data = tile_ds.ReadAsArray()  # 读取影像数据
#
#             tile_height = tile_ds.RasterYSize
#             tile_width = tile_ds.RasterXSize
#
#             # 计算放置位置
#             y_start = j * block_size
#             x_start = i * block_size
#
#             # 在拼接图像中放置裁剪块
#             stitched_image[:, y_start:y_start + tile_height, x_start:x_start + tile_width] = tile_data
#
#             tile_count += 1
#
#     # 更新输出图像的元数据
#     driver = gdal.GetDriverByName("GTiff")
#     out_ds = driver.Create(output_file, img_width, img_height, bands_count, gdal.GDT_Byte)
#
#     for band in range(1, bands_count + 1):
#         out_band = out_ds.GetRasterBand(band)
#         out_band.WriteArray(stitched_image[band - 1])
#
#     # 设置地理变换和投影
#     transform = original_ds.GetGeoTransform()
#     out_ds.SetGeoTransform(transform)
#     out_ds.SetProjection(original_ds.GetProjection())
#
#     out_ds.FlushCache()
#     print(f"合并完成，输出影像保存至: {output_file}")
#
#
# # 示例用法
# stitch_images(
#     input_dir=r'E:\lq\mmsegmentation-5channel\data\building_datasets-UVA\pre',  # 已经裁剪好的影像块文件夹
#     output_file=r'E:\MODEL\hebing\beit-5.tif',  # 输出合并影像的路径
#     original_image_path=r"E:\lq\mmsegmentation-5channel\data\building_datasets-UVA\1\1.tif"  # 原始影像路径，用于获取元数据
# )

from PIL import Image
import os
import numpy as np

# 输入文件夹路径和输出路径
input_folder = r'E:\lq\mmsegmentation-5channel\data\building_datasets-UVA\pre'  # 替换为你的影像文件夹路径
output_path = r'E:\MODEL\hebing\segnext-111.tif'  # 替换为你希望保存拼接结果的路径

# 获取所有影像文件的路径，按数字顺序排序
image_files = [f"{i}.tif" for i in range(3192)]  # 假设影像的命名方式是 0.tif, 1.tif, ..., 3191.tif
image_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字顺序排序

# 打开第一张图像以获取图像大小
first_image = Image.open(os.path.join(input_folder, image_files[0]))
image_width, image_height = first_image.size

# 创建一个空白的大图像，大小为 56 列 * 每张图像的宽度 和 57 行 * 每张图像的高度
merged_width = 56 * image_width
merged_height = 57 * image_height
merged_image = Image.new('L', (merged_width, merged_height))  # 使用'L'模式表示单通道灰度图像

# 拼接影像
for row in range(57):
    for col in range(56):
        # 计算当前影像的索引
        image_index = row * 56 + col
        if image_index >= len(image_files):  # 防止超出索引
            break
        image_path = os.path.join(input_folder, image_files[image_index])
        img = Image.open(image_path).convert('L')  # 确保是单通道（灰度图像）

        # 计算当前影像的拼接位置
        x_offset = col * image_width
        y_offset = row * image_height
        merged_image.paste(img, (x_offset, y_offset))

# 保存拼接结果
merged_image.save(output_path)
print(f"拼接后的影像已保存到 {output_path}")
