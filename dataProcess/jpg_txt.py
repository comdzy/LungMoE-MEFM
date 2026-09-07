import os
import numpy as np
import pydicom
from PIL import Image
import gc

# 路径配置
big_folder = r'D:\CTData\lung\LIDC\LIDC-IDRI'
txt_folder = r'D:\CTData\lung\Dataset\txt'
output_folder = r'D:\CTData\lung\Dataset\jpg'

# 创建输出文件夹（确保权限）
os.makedirs(output_folder, exist_ok=True)

def JPG_make(big_folder, txt_folder, output_folder):
    for root, _, files in os.walk(big_folder):
        for dicom_file in files:
            if not dicom_file.endswith('.dcm'):
                continue

            dicom_path = os.path.join(root, dicom_file)
            try:
                ds = pydicom.dcmread(dicom_path)
                sop_instance_uid = ds.SOPInstanceUID

                txt_file = os.path.join(txt_folder, f"{sop_instance_uid}.txt")
                if not os.path.exists(txt_file):
                    continue

                # 检查图像尺寸
                image_array = ds.pixel_array.astype(float)
                if image_array.shape != (512, 512):
                    continue

                # 解析窗宽/窗位
                window_center = ds.get("WindowCenter", 0)
                window_width = ds.get("WindowWidth", 1)  # 默认设为1避免除零

                # 处理多值属性
                if isinstance(window_center, (list, pydicom.multival.MultiValue)):
                    window_center = window_center[0] if len(window_center) > 0 else 0
                if isinstance(window_width, (list, pydicom.multival.MultiValue)):
                    window_width = window_width[0] if len(window_width) > 0 else 1

                if window_width == 0:
                    window_width = 1  # 防止除零

                # 图像归一化
                image_array = (image_array - window_center) / window_width
                image_array -= image_array.min()
                max_val = image_array.max()
                if max_val != 0:
                    image_array /= max_val
                else:
                    image_array = np.zeros_like(image_array)

                # 保存为 JPG
                image = Image.fromarray(np.uint8(image_array * 255), 'L')
                jpg_path = os.path.join(output_folder, f"{sop_instance_uid}.jpg")
                image.save(jpg_path)
                print(f"Saved {sop_instance_uid}.jpg")

            except Exception as e:
                print(f"Error processing {dicom_file}: {e}")
            finally:
                # 显式释放内存
                if 'ds' in locals():
                    del ds
                if 'image_array' in locals():
                    del image_array
                if 'image' in locals():
                    del image
                gc.collect()

    print("Conversion complete.")

JPG_make(big_folder, txt_folder, output_folder)