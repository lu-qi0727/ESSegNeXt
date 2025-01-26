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
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):  # 处理常见图像格式
            img_path = os.path.join(img_dir, filename)
            result = inference_model(model, img_path)

            # 假设 result.pred_sem_seg.data 是你要处理的张量，位于 GPU 上
            # 确保张量是正确的形状，并且是图像处理库所期望的形状，例如 (C, H, W) 或 (H, W)

            # 确定阈值，进行二值化操作
            threshold = 0.5
            binary_mask = (result.pred_sem_seg.data[0] > threshold).float()  # 处理第一个通道

            # 将张量从 GPU 移动到 CPU
            binary_mask = binary_mask.cpu()

            # 确保张量是单波段的，并且是正确的形状
            binary_mask = binary_mask.squeeze(0)  # 如果张量是单波段的，移除单维度

            # 将张量转换为 uint8 类型，乘以 255
            binary_mask = (binary_mask * 255).byte()  # 只需将二进制掩码转换为 uint8 类型

            # 使用 ToPILImage 转换张量为 PIL 图像
            pil_image = ToPILImage()(binary_mask)

            # 指定保存图片的文件夹路径
            save_path = Path(r'D:\lq\预测\ori\segnext1')

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
    parser.add_argument('-img_dir', default=r'D:\lq\predict_input\image', help='Directory containing input images')
    parser.add_argument('-config', default=r'E:\lq\mmsegmentation-3channel\configs\segnext\segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py', help='Config file')
    parser.add_argument('-checkpoint', default=r"D:\lq\训练结果\ori\segnext\iter_37200.pth", help='Checkpoint file')
    parser.add_argument('-out_dir', default='D:/lq/lq/noresult', help='Directory to save output images')
    parser.add_argument('-device', default='cuda:0', help='Device used for inference')
    parser.add_argument('-opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
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
