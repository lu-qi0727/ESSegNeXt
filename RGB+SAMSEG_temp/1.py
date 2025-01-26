import torch
from PIL import Image
import numpy as np
import os
from segment_anything import SamPredictor, sam_model_registry

# 模型路径和设备设置
model_path = r'E:\lq\mmsegmentation-main\my_model\mod_cls_txt_encoding.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(sam_model_registry.keys())

# 加载 SAM 模型
sam = sam_model_registry['vit_b'](checkpoint=model_path)
# print(sam_model_registry.keys())
sam.to(device)
predictor = SamPredictor(sam)


def generate_and_save_masks(image_dir, mask_dir):
    # 确保掩码文件夹存在
    os.makedirs(mask_dir, exist_ok=True)

    # 遍历图像文件夹
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # 运行 SAM 模型生成掩码
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=False  # 只输出单个掩码
        )

        # 保存掩码
        mask = masks[0]  # 选择第一个掩码
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # 转换为图像
        mask_save_path = os.path.join(mask_dir, image_name.replace('.png', '.png'))
        mask_image.save(mask_save_path)
        print(f"Saved mask for {image_name} to {mask_save_path}")
#
#
# 生成训练集掩码
train_image_dir = 'E:/lq/mmsegmentation-main/data/building_datasets/images/training'
train_mask_dir = r'C:\Users\admin\Desktop\lq-liter\vit_b_train'
generate_and_save_masks(train_image_dir, train_mask_dir)

# 生成验证集掩码
val_image_dir = 'E:/lq/mmsegmentation-main/data/building_datasets/images/validation'
val_mask_dir = r'C:\Users\admin\Desktop\lq-liter\vit_b_vali'
generate_and_save_masks(val_image_dir, val_mask_dir)
#
# import os
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset
#
#
# class BuildingDatasetWithMask(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         """
#         Args:
#             image_dir (str): 原始图像文件夹路径
#             mask_dir (str): 掩码文件夹路径（由 SAM 生成）
#             transform (callable, optional): 对图像和掩码进行的变换操作
#         """
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = os.listdir(image_dir)
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image_name = self.images[idx]
#         image_path = os.path.join(self.image_dir, image_name)
#         mask_path = os.path.join(self.mask_dir, image_name.replace('.png', '_mask.png'))
#
#         # 加载原始图像和掩码图像
#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")  # 掩码为单通道
#
#         # 将掩码转换为 NumPy 数组，并扩展为一个通道
#         mask_np = np.array(mask)
#         mask_np = np.expand_dims(mask_np, axis=-1)
#
#         # 将原始图像和掩码拼接，形成4通道输入（RGB + 掩码）
#         image_np = np.array(image)
#         combined_image = np.concatenate((image_np, mask_np), axis=-1)
#
#         # 将拼接后的图像转换为 PIL 图像
#         combined_image = Image.fromarray(combined_image)
#
#         if self.transform:
#             combined_image = self.transform(combined_image)
#
#         return combined_image
#
#





# # SegNext 网络配置



# _base_ = [
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_test.py',
#     '../_base_/datasets/building.py'
# ]
#
# ham_norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
# crop_size = (512, 512)
#
# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53, 0.0],  # 4 通道的均值（RGB + 掩码）
#     std=[58.395, 57.12, 57.375, 1.0],    # 4 通道的标准差（RGB + 掩码）
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255,
#     size=(512, 512),
#     test_cfg=dict(size_divisor=32))
#
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     pretrained=None,
#     backbone=dict(
#         type='MSCAN',
#         init_cfg=dict(),
#         embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
#         attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         in_channels=4  # 修改为 4 通道输入（RGB + 掩码）
#     ),
#     decode_head=dict(
#         type='LightHamHead',
#         in_channels=[64, 160, 256],
#         in_index=[1, 2, 3],
#         channels=256,
#         ham_channels=256,
#         dropout_ratio=0.1,
#         num_classes=2,
#         norm_cfg=ham_norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#         ham_kwargs=dict(
#             MD_S=1,
#             MD_R=16,
#             train_steps=6,
#             eval_steps=7,
#             inv_t=100,
#             rand_init=True)),
#     # model training and testing settings
#     train_cfg=dict(
#         type='IterBasedTrainLoop', max_iters=2000, val_interval=2000),
#     test_cfg=dict(mode='whole')
# )
#
# # 数据集配置
# train_dataloader = dict(batch_size=4)
#
# # 优化器配置
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=10.)
#         }))
#
# # 学习率调度器
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=810),
#     dict(
#         type='PolyLR',
#         power=1.0,
#         begin=0,
#         end=8100,
#         eta_min=0.0,
#         by_epoch=False,
#     )
# ]
