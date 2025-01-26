import os
import numpy as np
from osgeo import gdal
import torch
from mmseg.apis import inference_model, init_model


config_file = r'E:\lq\mmsegmentation-5channel\configs\segnext\segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py'  # 你训练模型的 config 文件路径
checkpoint_file = r"D:\lq\训练结果\geosam\segnext-bdloss\79.3.pth"# 你训练模型的 checkpoint 文件路径

input_folder = r'E:\lq\mmsegmentation-5channel\data\building_datasets5tongdao\images\training'

output_folder = r'E:\lq\mmsegmentation-3channel-order\data\building_datasets\pre-segnext-5'  # 保存预测结果的文件夹路径

# 初始化模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 定义阈值
threshold = 0.5

# 读取文件夹中的所有图像并进行预测
for image_file in os.listdir(input_folder):
    if image_file.endswith('.tif'):
        image_path = os.path.join(input_folder, image_file)

        # 使用 GDAL 读取五通道 TIFF 图像
        dataset = gdal.Open(image_path)
        img = dataset.ReadAsArray()
        img = np.transpose(img, (1, 2, 0))  # 转换为 (H, W, 5) 格式

        # 调试信息：输出图像的文件名和形状
        print(f"Processing image: {image_file}")
        print(f"Image shape: {img.shape}")

        # # 图像归一化到 0-1 范围
        # img = img.astype(np.float32) / 255.0

        # 调试信息：查看归一化后的图像范围
        print(f"Image min value: {np.min(img)}, max value: {np.max(img)}")

        # 将 numpy 数组转换为 list 以适应 mmsegmentation 输入要求
        result = inference_model(model, [img])

        # 调试信息：查看推理结果
        print(f"Inference result for {image_file}: {result}")

        # 提取预测结果 (SegDataSample 转换为 numpy)
        seg_pred = result[0].pred_sem_seg.data.cpu().numpy()
        seg_pred = seg_pred.squeeze()  # 去掉多余的维度 (H, W)

        # 调试信息：输出预测结果的形状和统计信息
        print(f"Prediction shape: {seg_pred.shape}")
        print(f"Prediction min value: {np.min(seg_pred)}, max value: {np.max(seg_pred)}")

        # 应用阈值，二值化预测结果
        binary_pred = (seg_pred <= threshold).astype(np.uint8)  # 大于等于阈值为 1，其他为 0

        # 调试信息：输出二值化后预测结果的统计信息
        print(
            f"Binary prediction after thresholding: min value: {np.min(binary_pred)}, max value: {np.max(binary_pred)}")

        # 将二值化后的预测结果转换为 uint8 类型
        pred = (binary_pred * 255).astype(np.uint8)

        # 保存预测结果为 TIFF
        output_path = os.path.join(output_folder, f'{image_file}')
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(output_path, pred.shape[1], pred.shape[0], 1, gdal.GDT_Byte)
        output_dataset.GetRasterBand(1).WriteArray(pred)
        output_dataset.FlushCache()

        # 调试信息：确认预测结果保存路径
        print(f"Prediction saved to: {output_path}")

print("预测完成，结果已保存至: ", output_folder)
