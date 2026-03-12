# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import re
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- 1. 【核心】完全匹配截图的视觉排序器 ---
def exact_visual_sort_key(filepath):
    """
    专门适配你截图的顺序：
    先按受试者编号排 (如 SC4001E0-PSG)
    再按 epoch 后的真实数字大小排 (如 305, 357...1041)
    """
    filename = os.path.basename(filepath)
    match = re.search(r"^(.*?)_epoch(\d+)_stage", filename)
    if match:
        base_name = match.group(1)       # 提取如 "SC4001E0-PSG"
        epoch_num = int(match.group(2))  # 提取 305, 1041 作为真正的数字比较大小
        return (base_name, epoch_num)
    # 如果遇到不符合规则的文件，退化为普通字符串排序
    return (filename, 0)

# --- 2. 基础模块定义 (保持不变) ---

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_projector = nn.Linear(512, 1024)

    def _make_layer(self, block, out_channels, blocks_num, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(blocks_num - 1):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_projector(x) 
        return x.unsqueeze(1) # [Batch, 1, 1024]

# --- 3. 特征提取、保存与日志逻辑 ---

def extract_and_save(model, image_folder, individual_save_folder, final_output_dir, final_output_name, device, transform):
    model.eval()
    all_features = []
    
    # 确保保存路径存在
    os.makedirs(individual_save_folder, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 【核心修改】获取文件列表，并使用我们写的 exact_visual_sort_key 严格排序
    raw_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(raw_files, key=exact_visual_sort_key)
    
    total_images = len(image_files)
    print(f"\n{'='*60}")
    print(f">>> 开始处理路径: {image_folder}")
    print(f">>> 文件夹内待处理图片总数: {total_images}")
    print(f"{'='*60}\n")

    processed_count = 0
    
    with torch.no_grad():
        for idx, img_name in enumerate(tqdm(image_files, desc="特征提取中")):
            img_path = os.path.join(image_folder, img_name)
            
            # 1. 读取图片
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # 2. 提取特征
            feature_tensor = model(input_tensor) # 形状: [1, 1, 1024]
            feature_numpy = feature_tensor.cpu().numpy()
            
            # 3. 验证提取结果
            status_text = "成功✅" if (feature_numpy is not None and feature_numpy.size > 0) else "失败❌"
            
            # 4. 【核心修改】每一次处理都打印出来，验证顺序是否正确
            # 使用 tqdm.write 可以在不打断进度条显示的情况下打印日志
            tqdm.write(f"[{idx+1}/{total_images}] 正在处理图片: {img_name} -> 特征提取: {status_text}")
            
            # 保存单张图片的 npy
            img_basename = os.path.splitext(img_name)[0]
            individual_save_path = os.path.join(individual_save_folder, f"{img_basename}.npy")
            np.save(individual_save_path, feature_numpy)
            
            all_features.append(feature_numpy)
            processed_count += 1

    # 合并为 [N, 1, 1024] 并保存汇总文件
    if all_features:
        final_features = np.concatenate(all_features, axis=0)
        final_save_path = os.path.join(final_output_dir, final_output_name)
        np.save(final_save_path, final_features)
        print(f"\n✅ 汇总特征保存成功: {final_save_path}")
        print(f"📊 汇总特征总形状: {final_features.shape} (N维度即为您刚看到的处理顺序)")
    
    print(f">>> 处理完成。成功提取特征图片数: {processed_count} / {total_images}\n")

# --- 4. 主程序 ---

if __name__ == "__main__":
    # 配置基础路径
    BASE_DATA_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event"
    
    # 图像源路径
    train_img_path = os.path.join(BASE_DATA_DIR, "train")
    eval_img_path = os.path.join(BASE_DATA_DIR, "test")
    
    # 特征npy保存路径 (每张图片)
    train_npy_save_path = os.path.join(BASE_DATA_DIR, "features_npy/train")
    eval_npy_save_path = os.path.join(BASE_DATA_DIR, "features_npy/test")
    
    # 汇总文件保存路径：eeg_vlm_sleep_data
    final_output_dir = os.path.join(BASE_DATA_DIR, "eeg_vlm_sleep_data")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. 处理训练集
    extract_and_save(model, train_img_path, train_npy_save_path, final_output_dir, "train_high_level_features.npy", device, transform)
    
    # 2. 处理验证集
    extract_and_save(model, eval_img_path, eval_npy_save_path, final_output_dir, "eval_high_level_features.npy", device, transform)

    print("\n🎉 所有特征提取任务已严格按【自然数字顺序】完成！")