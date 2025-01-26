# from samgeo import SamGeo
# from samgeo.text_sam import LangSAM
#
# image = r"F:\clip\image\63.png"
# # sam = SamGeo(
# #     model_type="vit_h",
# #     sam_kwargs=None,
# # )
#
#
# sam = SamGeo(
#     model_type="vit_h",
#     checkpoint=r"E:\lq\mmsegmentation-main\my_model\sam_vit_h_4b8939.pth",
#     automatic=True,
#     sam_kwargs=None,
# )
#
# sam.generate(image, output="masks.tif", foreground=True, unique=True)
# sam.raster_to_vector("masks.tif", "masks.shp")
#
#
# from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff
#
#
# #
# #
# import torch
# from samgeo import SamGeo
# # from samgeo.text_sam import LangSAM
#
# # 图像路径
# image = r"D:\lq\mmseg-test\mask\input\110.png"
#
# # 模型检查点路径
# checkpoint = r"D:\lq\mmseg-test\my_model\sam_vit_h_4b8939.pth"
#
# # 加载模型
# # sam_model = torch.load(checkpoint)  # 此处应该直接加载文件，而不是元组
#
# # 初始化 LangSAM
# # sam = LangSAM(checkpoint=checkpoint)
#
# # 设定文本提示
# # text_prompt = "tree"
#
# # 初始化 SamGeo
# sam_geo = SamGeo(
#     model_type="vit_h",
#     checkpoint=checkpoint,
#     automatic=True,
#     sam_kwargs=None,
# )
#
# # 生成掩码
# sam_geo.generate(image, output='D:\lq\mmseg-test\masks1.tif', foreground=True, unique=True)
#
# # 将栅格数据转换为矢量数据
# sam_geo.raster_to_vector("D:\lq\mmseg-test\masks1.tif", "D:\lq\mmseg-test\masks1.shp")
#
#
#
#
#
# import os
# import torch
# import cv2  # 用于图像读取和处理
# import numpy as np
# from samgeo import SamGeo
# # from samgeo.text_sam import LangSAM
#
# # 原始图像文件夹路径
# image_folder = r"F:\new\all\image"
# # 掩码输出文件夹路径
# mask_output_folder = r"F:\new\all\vit_l\l_output"
# # 四通道图像输出文件夹路径
# output_4channel_folder = r"F:\new\all\vit_l\l_4tongdao"
#
# # 模型检查点路径
# checkpoint = r"D:\lq\mmseg-test\my_model\sam_vit_l_0b3195.pth"
#
# # 加载模型
# sam_model = torch.load(checkpoint)
#
# # 初始化 LangSAM
# # sam = LangSAM(checkpoint=checkpoint)
#
# # 初始化 SamGeo
# sam_geo = SamGeo(
#     model_type="vit_l",
#     checkpoint=checkpoint,
#     automatic=True,
#     sam_kwargs=None,
# )
#
# # 创建输出文件夹（如果不存在）
# os.makedirs(mask_output_folder, exist_ok=True)
# os.makedirs(output_4channel_folder, exist_ok=True)
#
# # 批量处理文件夹中的图像
# for filename in os.listdir(image_folder):
#     if filename.endswith(".png") or filename.endswith(".jpg"):
#         # 图像的完整路径
#         image_path = os.path.join(image_folder, filename)
#
#         # 掩码的保存路径
#         mask_output_path = os.path.join(mask_output_folder, filename)
#
#         # 生成掩码
#         sam_geo.generate(image_path, output=mask_output_path, foreground=True, unique=True)
#
#         # 读取生成的掩码和原始图像
#         mask = cv2.imread(mask_output_path, cv2.IMREAD_GRAYSCALE)  # 单通道掩码
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取原始三通道图像
#
#         # 检查图像和掩码的大小是否一致
#         if image.shape[:2] != mask.shape[:2]:
#             print(f"尺寸不匹配: {filename}, 跳过...")
#             continue
#
#         # 将掩码扩展为一维通道，并与原图像拼接成四通道图像
#         mask_expanded = np.expand_dims(mask, axis=2)
#         image_4channel = np.concatenate((image, mask_expanded), axis=2)  # 拼接为4通道
#
#         # 保存4通道图像
#         output_4channel_path = os.path.join(output_4channel_folder, filename)
#         cv2.imwrite(output_4channel_path, image_4channel)
#
#         print(f"处理完成: {filename}")
#
# print("所有图像处理完成！")


import os
from samgeo import SamGeo

# Input and output paths
input_folder = r"F:\new\all\image_tiff"
mask_output_folder = r"F:\new\all\vit_h\h_output"
shp_output_folder = r"F:\new\all\vit_h\h_shp"

# Model checkpoint path
checkpoint = r"D:\lq\mmseg-test\my_model\sam_vit_h_4b8939.pth"

# Initialize SamGeo
sam_geo = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    automatic=True,
    sam_kwargs=None,
)

# Ensure output directories exist
os.makedirs(mask_output_folder, exist_ok=True)
os.makedirs(shp_output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        input_image = os.path.join(input_folder, filename)

        # Generate mask output path
        mask_output = os.path.join(mask_output_folder, f"{os.path.splitext(filename)[0]}_mask.tif")

        # Generate corresponding shapefile output path
        shp_output = os.path.join(shp_output_folder, f"{os.path.splitext(filename)[0]}.shp")

        # Generate mask
        sam_geo.generate(input_image, output=mask_output, foreground=True, unique=True)

        # Convert raster to vector shapefile
        sam_geo.raster_to_vector(mask_output, shp_output)

        print(f"Processed: {filename}")
