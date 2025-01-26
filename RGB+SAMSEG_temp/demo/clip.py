from osgeo import gdal
import os
import numpy as np


def merge_tiles_to_image(input_dir, output_path, tile_size, x_tiles, y_tiles, band_count):
    # 获取单个图块的地理变换和投影信息
    sample_tile_path = os.path.join(input_dir, 'tile_0_0.tif')
    sample_dataset = gdal.Open(sample_tile_path)
    geotransform = sample_dataset.GetGeoTransform()
    projection = sample_dataset.GetProjection()

    # 计算输出影像的尺寸
    x_max = x_tiles * tile_size
    y_max = y_tiles * tile_size

    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, x_max, y_max, band_count, gdal.GDT_UInt16)

    # 设置地理变换和投影
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)

    # 读取并合并图块
    for i in range(x_tiles):
        for j in range(y_tiles):
            # 构建当前图块的路径
            tile_path = os.path.join(input_dir, f'tile_{i}_{j}.tif')
            tile_dataset = gdal.Open(tile_path)
            tile_data = tile_dataset.GetRasterBand(1).ReadAsArray()

            # 计算图块在输出影像中的位置
            x_offset = i * tile_size
            y_offset = j * tile_size

            # 写入图块数据到输出数据集
            for band_index in range(1, band_count + 1):
                output_band = output_dataset.GetRasterBand(band_index)
                output_band.WriteArray(tile_data, x_offset, y_offset)

            # 清除图块数据集对象
            tile_dataset = None

    # 清除输出数据集对象缓存
    output_dataset.FlushCache()
    output_dataset = None

    print(f"合并完成，影像保存在: {output_path}")


# 示例调用
input_dir = r'D:\lq\JILIN\crop\prediction_result_4bandmss'
output_path = r'D:\lq\JILIN\hebing\result_4band_mss.tif'
tile_size = 512
x_tiles = 14  # 假设有10个图块沿X轴
y_tiles = 15  # 假设有10个图块沿Y轴
band_count = 4  # 假设有4个波段

merge_tiles_to_image(input_dir, output_path, tile_size, x_tiles, y_tiles, band_count)