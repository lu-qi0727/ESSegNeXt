import os
import cv2

# 输入文件夹路径
input_folder = r"E:\lq\mmsegmentation-3channel-order\data\building_datasets\pre-segnext-5"
# 输出文件夹路径
output_folder = os.path.join(input_folder, "inverted")

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # 检查文件是否为图像
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        # 读取图像
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 反转黑白颜色
            inverted_img = cv2.bitwise_not(img)
            # 保存反转后的图像
            cv2.imwrite(output_path, inverted_img)
            print(f"Inverted: {file_name}")
        else:
            print(f"Failed to read: {file_name}")
