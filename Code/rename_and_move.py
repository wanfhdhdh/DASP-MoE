import os
import shutil
import sys

def main():
    # 1. 定义源文件夹和目标文件夹路径
    # 注意：AutoDL 环境下通常是绝对路径 /root/...
    source_dir = "/root/autodl-tmp/MM-BSN/output/DEBRIS_Train_RN_Result"
    target_dir = "/root/autodl-tmp/MM-BSN/dataset/prep/DEBRIS_Train/RN_DN"

    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"[Error] 源文件夹不存在: {source_dir}")
        return

    # 2. 如果目标文件夹不存在，则自动创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[Info] 已创建目标文件夹: {target_dir}")
    else:
        print(f"[Info] 目标文件夹已存在: {target_dir}")

    # 获取源文件夹下的所有文件
    files = os.listdir(source_dir)
    count = 0
    
    print(f"正在处理... 共发现 {len(files)} 个文件")

    # 3. 遍历并处理
    for filename in files:
        # 只处理包含 _DN 的文件
        if "_DN" in filename:
            # 构建完整源路径
            src_file_path = os.path.join(source_dir, filename)
            
            # 确保是文件而不是文件夹
            if os.path.isfile(src_file_path):
                # 去掉文件名中的 _DN
                # 假设文件名格式为 image_name_DN.png -> image_name.png
                # 使用 replace 替换 '_DN.' 为 '.' 以保留后缀名
                new_filename = filename.replace("_DN.", ".")
                
                # 构建完整目标路径
                dst_file_path = os.path.join(target_dir, new_filename)
                
                # 复制并重命名 (使用 copy2 可以保留文件元数据如时间戳)
                shutil.copy2(src_file_path, dst_file_path)
                
                count += 1
                
                # 每处理100张打印一次进度
                if count % 100 == 0:
                    print(f"已处理 {count} 张图片...")

    print("-" * 30)
    print(f"[Success] 处理完成！")
    print(f"共复制并重命名了 {count} 张图片。")
    print(f"源路径: {source_dir}")
    print(f"目标路径: {target_dir}")
    print("-" * 30)

if __name__ == "__main__":
    main()