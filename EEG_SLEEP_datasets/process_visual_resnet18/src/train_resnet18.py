# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- 1. 排序器与数据集 ---

def exact_visual_sort_key(filepath):
    """
    专门适配截图顺序：先按受试者编号排，再按 epoch 后的真实数字大小排
    """
    filename = os.path.basename(filepath)
    match = re.search(r"^(.*?)_epoch(\d+)_stage", filename)
    if match:
        base_name = match.group(1)       
        epoch_num = int(match.group(2))  
        return (base_name, epoch_num)
    return (filename, 0)

class SleepEEGDataset(Dataset):
    """用于训练的 Dataset，负责读取图片并从文件名解析 Label"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
        
        # 假设标签映射：W->0, 1->1, 2->2, 3->3, R->4 (请根据你实际的文件名后缀调整)
        self.stage_map = {'W': 0, '1': 1, '2': 2, '3': 3, 'R': 4} 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 从文件名解析标签 (如 SC4001E0-PSG_epoch305_stage2.png -> 提取 '2')
        match = re.search(r"_stage([W1234R])", img_name)
        if match:
            stage_str = match.group(1)
            label = self.stage_map.get(stage_str, 0)
        else:
            label = 0 # 默认值
            
        return image, label

# --- 2. 基础模块定义 ---

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=5): 
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
        
        # 新增：分类头，用于训练算 Loss
        self.classifier = nn.Linear(1024, num_classes)

    def _make_layer(self, block, out_channels, blocks_num, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(blocks_num - 1):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, extract_features=False): 
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.feature_projector(x) 
        
        if extract_features:
            # 如果是提取特征模式，返回 1024 维特征 [Batch, 1, 1024]
            return features.unsqueeze(1)
            
        # 如果是训练模式，过分类头返回 logits
        return self.classifier(features)

# --- 3. 训练与特征提取逻辑 ---

def train_model(model, train_folder, device, transform, epochs=30, batch_size=8, lr=5e-4):
    """视觉增强模块独立训练：计算交叉熵损失并更新权重"""
    print(f"\n{'='*60}")
    print(f">>> 开始视觉增强模块训练...")
    print(f">>> 配置: Epochs={epochs}, Batch Size={batch_size}, LR={lr}")
    print(f"{'='*60}\n")
    
    dataset = SleepEEGDataset(train_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, extract_features=False) # 训练模式
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100*correct/total:.1f}%"})
            
    print("\n✅ 视觉增强模块独立训练完成，特征提取权重已更新！\n")

def extract_and_save(model, image_folder, individual_save_folder, final_output_dir, final_output_name, device, transform):
    """保持原样的严格顺序特征提取（使用已训练的权重）"""
    model.eval()
    all_features = []
    
    os.makedirs(individual_save_folder, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    raw_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(raw_files, key=exact_visual_sort_key)
    
    total_images = len(image_files)
    print(f"\n{'='*60}")
    print(f">>> 开始严格有序特征提取: {image_folder}")
    print(f">>> 待处理图片总数: {total_images}")
    print(f"{'='*60}\n")

    processed_count = 0
    with torch.no_grad():
        for idx, img_name in enumerate(tqdm(image_files, desc="特征提取与保存中")):
            img_path = os.path.join(image_folder, img_name)
            
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # 【核心】：指定 extract_features=True 获取 1024 维特征，而不经过分类器
            feature_tensor = model(input_tensor, extract_features=True) 
            feature_numpy = feature_tensor.cpu().numpy()
            
            status_text = "成功✅" if (feature_numpy is not None and feature_numpy.size > 0) else "失败❌"
            tqdm.write(f"[{idx+1}/{total_images}] 处理图片: {img_name} -> {status_text}")
            
            # 1. 保存单张图片的 npy
            img_basename = os.path.splitext(img_name)[0]
            individual_save_path = os.path.join(individual_save_folder, f"{img_basename}.npy")
            np.save(individual_save_path, feature_numpy)
            
            all_features.append(feature_numpy)
            processed_count += 1

    # 2. 合并并保存汇总文件
    if all_features:
        final_features = np.concatenate(all_features, axis=0)
        final_save_path = os.path.join(final_output_dir, final_output_name)
        
        # 👉 【加上这行救命的代码，强行创建文件夹！】
        os.makedirs(final_output_dir, exist_ok=True) 
        
        np.save(final_save_path, final_features)
        print(f"\n✅ 汇总特征保存成功: {final_save_path}")

# --- 4. 主程序入口 ---

if __name__ == "__main__":
    # 配置基础路径
    BASE_DATA_DIR = "/mnt/inaisfs/workspace/EEG-VLM/data/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event"
    
    # 图片源路径
    train_img_path = os.path.join(BASE_DATA_DIR, "train")
    eval_img_path = os.path.join(BASE_DATA_DIR, "test")
    
    # 单个特征保存路径
    train_npy_save_path = os.path.join(BASE_DATA_DIR, "features_npy/train")
    eval_npy_save_path = os.path.join(BASE_DATA_DIR, "features_npy/test")
    
    # 汇总特征保存路径
    final_output_dir = os.path.join(BASE_DATA_DIR, "eeg_vlm_sleep_data")
    
    # 设备与模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=5).to(device)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ==========================================
    # 步骤一：视觉增强模块独立训练 (学习特征表达)
    # ==========================================
    train_model(model, train_img_path, device, transform, epochs=30, batch_size=8, lr=5e-4)
    
    # 【修复安全隐患】保存训练好的权重前，确保 checkpoint 文件夹存在
    checkpoint_dir = os.path.join(BASE_DATA_DIR, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_weight_path = os.path.join(checkpoint_dir, "resnet18_visual_enhanced_trained.pth")
    torch.save(model.state_dict(), model_weight_path)
    print(f"💾 模型权重已保存至: {model_weight_path}")

    # ==========================================
    # 步骤二：使用训练好的权重，严格按顺序提取特征
    # ==========================================
    extract_and_save(model, train_img_path, train_npy_save_path, final_output_dir, "train_high_level_features.npy", device, transform)
    extract_and_save(model, eval_img_path, eval_npy_save_path, final_output_dir, "eval_high_level_features.npy", device, transform)

    print("\n🎉 全部任务（视觉增强模块训练 + 严格有序特征提取）已圆满完成！")