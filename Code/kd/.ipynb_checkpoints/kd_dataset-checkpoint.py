import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SonarDataset(Dataset):
    def __init__(self, input_dir, label_dir=None, image_size=(160, 160), transform=None, stage=1):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.stage = stage
        
        # 获取所有图片文件名 (过滤非图片文件)
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if stage == 2:
            assert self.label_dir is not None, "Stage 2 requires a label directory (Teacher output)"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # 1. 读取输入 (噪声图)
        input_path = os.path.join(self.input_dir, filename)
        image = Image.open(input_path).convert('L') # 强制转为灰度图
        image = image.resize(self.image_size)
        
        if self.transform:
            image = self.transform(image)

        # 2. 读取标签 (Teacher 去噪图)
        if self.stage == 2:
            # 假设 Teacher 输出的文件名和输入一致，或者有后缀
            # 这里需要根据你的 generate_pseudo_labels.py 生成规则来定
            # 如果文件名完全一致：
            label_path = os.path.join(self.label_dir, filename)
            
            # 如果文件名有后缀 (例如 _DN.png):
            # label_name = filename.rsplit('.', 1)[0] + '_DN.' + filename.rsplit('.', 1)[1]
            # label_path = os.path.join(self.label_dir, label_name)
            
            label = Image.open(label_path).convert('L')
            label = label.resize(self.image_size)
            if self.transform:
                label = self.transform(label)
            return image, label
        else:
            # Stage 1: 自重构，标签就是输入自己
            return image, image