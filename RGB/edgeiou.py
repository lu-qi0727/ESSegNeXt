import os
import cv2
import numpy as np


def calculate_edge_iou(pred_edge, gt_edge):
    """
    计算边缘IoU
    :param pred_edge: 预测图的边缘二值图
    :param gt_edge: 真值图的边缘二值图
    :return: 边缘IoU
    """
    intersection = np.logical_and(pred_edge, gt_edge).sum()
    union = np.logical_or(pred_edge, gt_edge).sum()
    if union == 0:  # 避免除零
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def extract_edge(image, kernel_size=3):
    """
    提取边缘
    :param image: 输入二值图像
    :param kernel_size: 用于膨胀和腐蚀的卷积核大小
    :return: 边缘二值图
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    eroded = cv2.erode(image, kernel, iterations=1)
    edge = cv2.absdiff(dilated, eroded)
    return edge


def process_images(pred_folder, gt_folder):
    """
    处理文件夹中的预测图和真实图，计算边缘IoU
    :param pred_folder: 预测图文件夹路径
    :param gt_folder: 真值图文件夹路径
    :return: 边缘IoU结果字典
    """
    pred_files = sorted(os.listdir(pred_folder))
    gt_files = sorted(os.listdir(gt_folder))

    results = {}

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, gt_file)

        # 读取图像并转换为二值图
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        _, pred_bin = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)
        _, gt_bin = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)

        # 提取边缘
        pred_edge = extract_edge(pred_bin)
        gt_edge = extract_edge(gt_bin)

        # 计算边缘IoU
        iou = calculate_edge_iou(pred_edge > 0, gt_edge > 0)
        results[pred_file] = iou

    return results
#
# pred_folder = r'E:\lq\mmsegmentation-3channel\data\building_datasets\pre'
# gt_folder = r'E:\lq\mmsegmentation-3channel\data\building_datasets\gt'

pred_folder = r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA\pre-upternet1'
gt_folder = r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA\gt'

# 计算边缘IoU
edge_iou_results = process_images(pred_folder, gt_folder)

# 输出结果并计算均值
total_iou = 0
count = 0
for file, iou in edge_iou_results.items():
    print(f"{file}: Edge IoU = {iou:.4f}")
    total_iou += iou
    count += 1

# 计算均值
if count > 0:
    mean_iou = total_iou / count
    print(f"\nMean Edge IoU = {mean_iou:.4f}")
else:
    print("\nNo files to process!")
