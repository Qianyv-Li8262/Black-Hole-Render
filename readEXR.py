import cv2
import numpy as np

def read_exr_with_opencv(file_path):
    # 使用 cv2.IMREAD_UNCHANGED 标志，以保持 32 位浮点通道数据
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)*50
    
    if image is None:
        print("无法读取文件，请检查路径是否正确。")
        return None
        
    print(f"图像尺寸 (高, 宽, 通道): {image.shape}")
    print(f"数据类型: {image.dtype}")  # 通常为 float32
    
    # 示例操作：由于 EXR 是高动态范围（HDR），直接保存为普通格式前
    # 建议进行简单的色调映射（Tone Mapping）或归一化，否则高光部分可能会过曝
    # 这里使用简单的裁剪和线性缩放作为演示：
    gamma_corrected = np.power(np.clip(image, 0, 1), 1.0 / 2.2) # 简易伽马校正
    output_image = (gamma_corrected * 255).astype(np.uint8)
    
    # 保存为普通格式以供预览
    cv2.imwrite("output_preview.png", output_image)
    print("预览图已保存为 output_preview.png")
    
    return image

# 替换为您的 EXR 文件路径
read_exr_with_opencv("starmap_random_2020_16k.exr")