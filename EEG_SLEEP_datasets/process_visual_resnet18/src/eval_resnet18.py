# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- 1. 严格顺序排序器 ---

def exact_visual_sort_key(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r"^(.*?)_epoch(\d+)", filename)
    if match:
        base_name = match.group(1)       
        epoch_num = int(match.group(2))  
        return (base_name, epoch_num)
    return (filename, 0)

# --- 2. 评估专用数据集 ---

class EvalSleepDataset(Dataset):
    def __init__(self, folder_path, transform=None, has_labels=True):
        self.folder_path = folder_path
        self.transform = transform
        self.has_labels = has_labels
        
        raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
        self.image_files = sorted(raw_files, key=exact_visual_sort_key)
        
        self.stage_map = {'W': 0, '1': 1, '2': 2, '3': 3, 'R': 4} 
        self.inv_stage_map = {0: 'W', 1: '1', 2: '2', 3: '3', 4: 'R'}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        if self.has_labels:
            match = re.search(r"_stage([W1234R])", img_name)
            label = self.stage_map.get(match.group(1), 0) if match else 0
            return image, label, img_name
        else:
            return image, img_name

# --- 3. 模型定义 ---

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
            return features.unsqueeze(1)
        return self.classifier(features)

# --- 4. 核心：直接评估 test1 并核对答案 ---

def evaluate_test1_vs_test(model, test1_folder, test_folder, device, transform):
    """直接预测 test1(无标签) 并通过读取 test(有标签) 当场计算成功率"""
    print(f"\n{'='*60}")
    print(f">>> 开始预测验证集: {test1_folder}")
    print(f">>> 对照标准答案集: {test_folder}")
    print(f"{'='*60}\n")
    
    # 1. 建立标准答案字典 (从 test 文件夹)
    true_labels = {}
    if not os.path.exists(test_folder):
        print(f"❌ 找不到带标签的标准答案文件夹: {test_folder}")
        return

    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg')):
            match = re.search(r"^(.*?)_epoch(\d+)_stage([W1234R])", filename)
            if match:
                base_name = match.group(1)
                epoch = int(match.group(2))
                true_stage = match.group(3)
                true_labels[(base_name, epoch)] = true_stage
                
    print(f"📥 成功获取了 {len(true_labels)} 个标准答案。")

    # 2. 对无标签的 test1 进行推理并当场对比
    dataset = EvalSleepDataset(test1_folder, transform=transform, has_labels=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = {stage: 0 for stage in ['W', '1', '2', '3', 'R']}
    class_total = {stage: 0 for stage in ['W', '1', '2', '3', 'R']}
    
    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc="正在预测并核对"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(img_names)):
                img_name = img_names[i]
                pred_idx = predicted[i].item()
                pred_stage = dataset.inv_stage_map[pred_idx]
                
                # 提取考题的编号
                match = re.search(r"^(.*?)_epoch(\d+)", img_name)
                if match:
                    base_name = match.group(1)
                    epoch = int(match.group(2))
                    
                    # 翻看标准答案
                    true_stage = true_labels.get((base_name, epoch))
                    if true_stage is not None:
                        total += 1
                        class_total[true_stage] += 1
                        if true_stage == pred_stage:
                            correct += 1
                            class_correct[true_stage] += 1

    # 3. 结算成绩单
    print(f"\n📝 批改完成，共核对 {total} 份答卷。")
    if total > 0:
        accuracy = 100 * correct / total
        print(f"\n✅ test1 无标签验证集 - 整体准确率: {accuracy:.2f}% ({correct}/{total})\n")
        print("📊 各睡眠阶段预测准确率:")
        for stage in ['W', '1', '2', '3', 'R']:
            if class_total[stage] > 0:
                acc = 100 * class_correct[stage] / class_total[stage]
                print(f"  - Stage {stage}: {acc:.2f}% ({class_correct[stage]}/{class_total[stage]})")
    else:
        print("⚠️ 未能匹配到任何对应的标签，请检查 test1 和 test 文件夹的文件名是否匹配。")

# --- 5. 主程序 ---

if __name__ == "__main__":
    BASE_DATA_DIR = "/mnt/inaisfs/workspace/EEG-VLM/data/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event"
    test_labeled_path = os.path.join(BASE_DATA_DIR, "test")
    test_unlabeled_path = os.path.join(BASE_DATA_DIR, "test1")
    WEIGHT_PATH = os.path.join(BASE_DATA_DIR, "checkpoint/resnet18_visual_enhanced_trained.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. 加载权重
    model = ResNet18(num_classes=5).to(device)
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH))
        print(f"成功加载训练好的权重: {WEIGHT_PATH}")
    else:
        print(f"❌ 错误: 找不到模型权重文件 {WEIGHT_PATH}，请检查路径或先运行训练代码！")
        exit()

    # 2. 一步到位：预测 test1 并核对 test
    if os.path.exists(test_unlabeled_path) and os.path.exists(test_labeled_path):
        evaluate_test1_vs_test(model, test_unlabeled_path, test_labeled_path, device, transform)
    else:
        print(f"❌ 请确保 test 和 test1 文件夹都存在！")