import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ==========================================
# 关键导入：确保 kd 文件夹在当前目录下
# ==========================================
from kd.vqvae import VQVAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",           type=str, required=True,    help="包含待去噪噪声图像的文件夹路径")
    parser.add_argument("--model_weight_dir",   type=str, required=True,    help="训练好的 VQ-VAE 权重文件路径 (.pt)")
    parser.add_argument("--output_dir",         type=str, default="output/results_kd", help="去噪结果保存路径")
    parser.add_argument("--gpu_id",             type=str, default='0',      help="使用的 GPU ID (例如 0)")
    args = parser.parse_args()

    # 1. 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[Info] 结果将保存在: {args.output_dir}')

    # 2. 设备设置
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] 使用设备: {device}')

    # 3. 加载模型
    # 注意：in_channel=1 对应声纳单通道灰度图
    model = VQVAE(in_channel=1).to(device)
    
    if not os.path.exists(args.model_weight_dir):
        raise FileNotFoundError(f"找不到权重文件: {args.model_weight_dir}")
        
    print(f'[Info] 正在加载权重: {args.model_weight_dir}')
    checkpoint = torch.load(args.model_weight_dir, map_location=device)
    
    # 兼容处理：有些权重保存时包含了 'state_dict' 键，有些直接是字典
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 4. 获取所有图片文件
    img_paths = []
    for root, dirs, files in os.walk(args.test_dir):
        # [修改 1] 这一行非常重要：原地修改 dirs 列表，
        # 告诉 os.walk 不要进入以 . 开头的隐藏文件夹（例如 .ipynb_checkpoints）
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            # [修改 2] 双重保险：如果文件名里包含 checkpoint，直接跳过
            if 'checkpoint' in file:
                continue

            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                # 只有当文件在根目录时才添加，防止路径拼接错误
                if root == args.test_dir:
                    img_paths.append(file)
    img_paths.sort()
    
    if len(img_paths) == 0:
        print("[Error] 测试文件夹为空或没有支持的图片格式！")
        return

    print(f'[Info] 发现 {len(img_paths)} 张图片，开始推理...')

    # 5. 推理循环
    with torch.no_grad():
        time_begin = time.time()
        for idx, img_name in enumerate(img_paths):
            # A. 读取图片 (灰度模式)
            full_path = os.path.join(args.test_dir, img_name)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"[Warning] 无法读取图片: {img_name}, 跳过。")
                continue

            # B. 预处理: numpy -> tensor, 增加维度 [H,W] -> [1, 1, H, W]
            img = img.astype(np.float32)
            noisy_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

            # C. Padding (关键步骤)
            # VQ-VAE 有下采样操作，输入尺寸必须是 8 的倍数
            b, c, h, w = noisy_img.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h != 0 or pad_w != 0:
                noisy_img = F.pad(noisy_img, (0, pad_w, 0, pad_h), mode='reflect') # 建议用 reflect 填充减少边缘伪影

            # D. 归一化 (0~255 -> -1~1)
            # 这必须与 train_KD.py 中的 transforms.Normalize([0.5], [0.5]) 保持一致
            # (x / 255 - 0.5) * 2  <==> x / 127.5 - 1
            noisy_input = (noisy_img / 255.0 - 0.5) * 2.0

            # E. 模型推理
            denoised_img, _ = model(noisy_input)

            # F. 反归一化 (-1~1 -> 0~255)
            denoised_img = (denoised_img / 2.0 + 0.5) * 255.0
            denoised_img = torch.clamp(denoised_img, 0, 255)

            # G. 裁剪掉 Padding 部分
            denoised_img = denoised_img[:, :, :h, :w]

            # H. 保存结果
            # Tensor [1, 1, H, W] -> Numpy [H, W]
            output = denoised_img.squeeze().cpu().numpy().astype(np.uint8)
            
            save_name = os.path.splitext(img_name)[0] + '_DN.png'
            cv2.imwrite(os.path.join(args.output_dir, save_name), output)

            if (idx + 1) % 10 == 0:
                print(f'Processed {idx + 1}/{len(img_paths)}')

        time_end = time.time()
        print(f'[Success] 全部完成！耗时: {time_end - time_begin:.2f}s')
        print(f'平均每张耗时: {(time_end - time_begin)/len(img_paths):.4f}s')

if __name__ == "__main__":
    main()