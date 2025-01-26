# from torchvision.transforms import ToPILImage
# from PIL import Image
# import numpy as np
# from pathlib import Path
# import os
# from argparse import ArgumentParser
#
# from mmengine.model import revert_sync_batchnorm
# from mmseg.apis import inference_model, init_model, show_result_pyplot
#
# def process_images(img_dir, config, checkpoint, out_dir=None, device='cuda:0', opacity=0.5, with_labels=False, title='result'):
#     # Initialize model
#     model = init_model(config, checkpoint, device=device)
#     if device == 'cpu':
#         model = revert_sync_batchnorm(model)
#
#     # Create output directory if not exists
#     if out_dir is not None and not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     # Process each image in the input directory
#     for filename in os.listdir(img_dir):
#         if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
#             img_path = os.path.join(img_dir, filename)
#             result = inference_model(model, img_path)
#             print(result.pred_sem_seg.data)
#
#             # 假设 result.pred_sem_seg.data 是你要处理的张量，位于 GPU 上
#             # 确保张量是正确的形状，并且是图像处理库所期望的形状，例如 (C, H, W) 或 (H, W)
#
#             # 确定阈值，进行二值化操作
#             threshold = 0.5
#             binary_mask = (result.pred_sem_seg.data > threshold).float()
#
#             # 将张量从 GPU 移动到 CPU
#             binary_mask = binary_mask.cpu()
#
#             # 确保张量是单波段的，并且是正确的形状
#             binary_mask = binary_mask.squeeze(0)  # 如果张量是单波段的，移除单维度
#
#             # 将张量转换为 torch.uint8 类型，并乘以 255
#             binary_mask = (binary_mask * 255).byte()
#
#             # 使用 ToPILImage 转换张量为 PIL 图像
#             pil_image = ToPILImage()(binary_mask)
#
#             # 指定保存图片的文件夹路径
#         save_path = Path(r'D:\lq\预测\geosam\segnext')
#         # save_path = Path('D:\lq\JILIN\prediction\segnext\prediction_result_mss_128')
#         # 确保文件夹存在，如果不存在则创建
#         save_path.mkdir(parents=True, exist_ok=True)
#
#         # 指定图片文件名和保存路径
#         image_filename = filename
#         image_path = save_path / image_filename
#
#         # 保存图像到指定路径
#         pil_image.save(image_path)
#
#         print(f'Binary image saved to: {image_path}')
#
#         out_file = None
#         if out_dir is not None:
#             out_file = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_result.png")
#
#         show_result_pyplot(
#             model,
#             img_path,
#             result,
#             title=title,
#             opacity=opacity,
#             with_labels=with_labels,
#             draw_gt=False,
#             show=False if out_dir is not None else True,
#             out_file=out_file)
#
#
#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-img_dir', default=r'F:\new\dataset\5_channel', help='Directory containing input images')
#
#     parser.add_argument('-config',default=r'configs/segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py',help='Config file')
#
#
#
#     parser.add_argument('-checkpoint', default=r"D:\lq\训练结果\geosam\5channel_segnext\iter_50000.pth",help='Checkpoint file')#有sam
#
#     # parser.add_argument('-checkpoint', default=r'E:\lq\mmsegmentation-main\work_dirs\segnext_jilin\iter_8100.pth',help='Checkpoint file')#无sam
#     # parser.add_argument('-checkpoint', default=r'E:\lq\mmsegmentation-main\work_dirs\segnext_lq\iter_56000.pth', help='Checkpoint file')
#
#
#     parser.add_argument('-out_dir', default='D:/lq/lq/noresult', help='Directory to save output images')
#     parser.add_argument('-device', default='cuda:0', help='Device used for inference')
#     parser.add_argument('-opacity', type=float, default=0.5,
#                         help='Opacity of painted segmentation map. In (0, 1] range.')
#     parser.add_argument('-with_labels', action='store_true', default=False, help='Whether to display the class labels.')
#     parser.add_argument('-title', default='result', help='The image identifier.')
#
#     args = parser.parse_args()
#
#     # Use args.img_dir to get the input directory
#     img_dir = args.img_dir
#     config = args.config
#     checkpoint = args.checkpoint
#     out_dir = args.out_dir
#     device = args.device
#     opacity = args.opacity
#     with_labels = args.with_labels
#     title = args.title
#
#     process_images(img_dir, config, checkpoint, out_dir, device, opacity, with_labels, title)

#
#
#
# #
# # #
# # #
# # #
# # #
# #
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
from pathlib import Path
import os
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot

def process_images(img_dir, config, checkpoint, out_dir=None, device='cuda:0', opacity=0.5, with_labels=False, title='result'):
    # Initialize model
    model = init_model(config, checkpoint, device=device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Create output directory if not exists
    if out_dir is not None and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process each image in the input directory
    for filename in os.listdir(img_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
            img_path = os.path.join(img_dir, filename)
            result = inference_model(model, img_path)
            print(result.pred_sem_seg.data)

            # 假设 result.pred_sem_seg.data 是你要处理的张量，位于 GPU 上
            # 确保张量是正确的形状，并且是图像处理库所期望的形状，例如 (C, H, W) 或 (H, W)

            # 确定阈值，进行二值化操作
            threshold = 0.5
            binary_mask = (result.pred_sem_seg.data > threshold).float()

            # 将张量从 GPU 移动到 CPU
            binary_mask = binary_mask.cpu()

            # 确保张量是单波段的，并且是正确的形状
            binary_mask = binary_mask.squeeze(0)  # 如果张量是单波段的，移除单维度

            # 将张量转换为 torch.uint8 类型，并乘以 255
            binary_mask = (binary_mask * 255).byte()

            # 使用 ToPILImage 转换张量为 PIL 图像
            pil_image = ToPILImage()(binary_mask)

            # 指定保存图片的文件夹路径
        save_path = Path(r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA\pre-upternet1')
        # save_path = Path('D:\lq\JILIN\prediction\segnext\prediction_result_mss_128')
        # 确保文件夹存在，如果不存在则创建
        save_path.mkdir(parents=True, exist_ok=True)

        # 指定图片文件名和保存路径
        image_filename = filename
        image_path = save_path / image_filename

        # 保存图像到指定路径
        pil_image.save(image_path)

        print(f'Binary image saved to: {image_path}')

        out_file = None
        if out_dir is not None:
            out_file = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_result.png")

        show_result_pyplot(
            model,
            img_path,
            result,
            title=title,
            opacity=opacity,
            with_labels=with_labels,
            draw_gt=False,
            show=False if out_dir is not None else True,
            out_file=out_file)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-img_dir', default=r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA\image', help='Directory containing input images')

    parser.add_argument('-config',default=r'E:\lq\mmsegmentation-3channel-order\configs\upernet\upernet_r18_4xb4-160k_ade20k-512x512.py',help='Config file')



    parser.add_argument('-checkpoint', default=r"E:\MODEL\ORI\upernet\73.31.pth",help='Checkpoint file')#有sam


    parser.add_argument('-out_dir', default='D:/lq/lq/noresult', help='Directory to save output images')
    parser.add_argument('-device', default='cuda:0', help='Device used for inference')
    parser.add_argument('-opacity', type=float, default=0.5,
                        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('-with_labels', action='store_true', default=False, help='Whether to display the class labels.')
    parser.add_argument('-title', default='result', help='The image identifier.')

    args = parser.parse_args()

    # Use args.img_dir to get the input directory
    img_dir = args.img_dir
    config = args.config
    checkpoint = args.checkpoint
    out_dir = args.out_dir
    device = args.device
    opacity = args.opacity
    with_labels = args.with_labels
    title = args.title

    process_images(img_dir, config, checkpoint, out_dir, device, opacity, with_labels, title)
# #
# #
# #
# #
# #
# #
# #
# #





#
# import os
# import numpy as np
# from osgeo import gdal
# import torch
# from mmseg.apis import inference_model, init_model
#
# # 配置文件和检查点文件路径
# config_file = r'E:\lq\mmsegmentation-3channel-order\configs\beit\beit-base_upernet_8xb2-160k_ade20k-640x640.py'  # 你训练模型的 config 文件路径
# checkpoint_file = r"E:\MODEL\ORI\beit\64.62.pth"  # 你训练模型的 checkpoint 文件路径
#
# input_folder = r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA\image'
# output_folder = r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA\pre'  # 保存预测结果的文件夹路径
#
# # 初始化模型
# model = init_model(config_file, checkpoint_file, device='cuda:0')
#
# # 创建输出文件夹
# os.makedirs(output_folder, exist_ok=True)
#
# # 定义阈值
# threshold = 0.5
#
# # 读取文件夹中的所有图像并进行预测
# for image_file in os.listdir(input_folder):
#     if image_file.endswith('.tif'):
#         image_path = os.path.join(input_folder, image_file)
#
#         # 使用 GDAL 读取 RGB 图像（假设图像的前 3 通道是 RGB）
#         dataset = gdal.Open(image_path)
#         img = dataset.ReadAsArray()
#
#         # 提取 RGB 三通道数据 (假设前三个通道为 RGB)
#         img = img[:3, :, :]  # 只选择前三个通道（RGB）
#
#         img = np.transpose(img, (1, 2, 0))  # 转换为 (H, W, 3) 格式
#
#         # 调试信息：输出图像的文件名和形状
#         print(f"Processing image: {image_file}")
#         print(f"Image shape: {img.shape}")
#
#         # # 图像归一化到 0-1 范围
#         # img = img.astype(np.float32) / 255.0
#
#         # 调试信息：查看归一化后的图像范围
#         print(f"Image min value: {np.min(img)}, max value: {np.max(img)}")
#
#         # 将 numpy 数组转换为 list 以适应 mmsegmentation 输入要求
#         result = inference_model(model, [img])
#
#         # 调试信息：查看推理结果
#         print(f"Inference result for {image_file}: {result}")
#
#         # 提取预测结果 (SegDataSample 转换为 numpy)
#         seg_pred = result[0].pred_sem_seg.data.cpu().numpy()
#         seg_pred = seg_pred.squeeze()  # 去掉多余的维度 (H, W)
#
#         # 调试信息：输出预测结果的形状和统计信息
#         print(f"Prediction shape: {seg_pred.shape}")
#         print(f"Prediction min value: {np.min(seg_pred)}, max value: {np.max(seg_pred)}")
#
#         # 应用阈值，二值化预测结果
#         binary_pred = (seg_pred >= threshold).astype(np.uint8)  # 大于等于阈值为 1，其他为 0
#
#         # 调试信息：输出二值化后预测结果的统计信息
#         print(
#             f"Binary prediction after thresholding: min value: {np.min(binary_pred)}, max value: {np.max(binary_pred)}")
#
#         # 将二值化后的预测结果转换为 uint8 类型
#         pred = (binary_pred * 255).astype(np.uint8)
#
#         # 保存预测结果为 TIFF
#         output_path = os.path.join(output_folder, f'{image_file}')
#         driver = gdal.GetDriverByName('GTiff')
#         output_dataset = driver.Create(output_path, pred.shape[1], pred.shape[0], 1, gdal.GDT_Byte)
#         output_dataset.GetRasterBand(1).WriteArray(pred)
#         output_dataset.FlushCache()
#
#         # 调试信息：确认预测结果保存路径
#         print(f"Prediction saved to: {output_path}")
#
# print("预测完成，结果已保存至: ", output_folder)

