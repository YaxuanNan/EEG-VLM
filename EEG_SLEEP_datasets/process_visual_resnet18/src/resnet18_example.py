import torch
import torch.nn as nn  # PyTorch的神经网络工具箱
 
class BasicBlock(nn.Module):
    # 初始化：定义残差块里的所有“小零件”
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 1. 第一层卷积：3×3卷积核，步长stride（控制特征图尺寸是否缩小）
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,  # 输入特征图的“通道数”（比如64个通道=64种特征）
            out_channels=out_channels, # 输出特征图的通道数
            kernel_size=3,             # 卷积核大小（3×3，提取局部特征）
            stride=stride,             # 步长（1=尺寸不变，2=尺寸缩小一半）
            padding=1,                 # 边缘填充（保证卷积后尺寸符合预期）
            bias=False                 # 因为后面有BN，BN会处理偏置，这里设为False
        )
        # 2. 第一层卷积后的批量归一化（BN）：让数据分布更稳定，加速训练
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3. ReLU激活函数：给模型加“非线性”，让它能学习复杂特征（比如从边缘学到纹理）
        self.relu = nn.ReLU(inplace=True)  # inplace=True：节省内存
        
        # 4. 第二层卷积：和第一层结构类似，但步长固定为1（不改变尺寸）
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 5. 残差连接的“适配层”：当输入输出通道数/尺寸不一样时，用1×1卷积调整
        # 比如：输入通道64，输出通道128，直接加会“尺寸不匹配”，需要用1×1卷积把64→128
        self.downsample = None  # 默认没有适配层（输入输出一致时）
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(  # 用“序列容器”把层打包
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    # 定义数据流动路径（forward=向前传播）
    def forward(self, x):
        residual = x  # 先把原始输入存起来（对应“残差连接的捷径”）
        
        # 1. 走“正常卷积路径”：conv1 → bn1 → relu → conv2 → bn2
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 2. 残差连接：如果需要适配，先调整原始输入的通道/尺寸，再和卷积结果相加
        if self.downsample is not None:
            residual = self.downsample(x)  # 适配原始输入
        out += residual  # 核心：卷积结果 + 原始输入（残差连接）
        
        # 3. 最后激活，输出该残差块的结果
        out = self.relu(out)
        return out
    
class ResNet18(nn.Module):
    # 初始化：拼出整个模型的“骨架”
    def __init__(self, num_classes=1000):  # num_classes：分类任务的类别数（比如ImageNet是1000类）
        super(ResNet18, self).__init__()
        # 1. 初始处理层：把输入图片（比如3通道RGB图）转成64通道特征图，同时缩小尺寸
        self.in_channels = 64  # 后续残差块的“输入通道数”初始值
        self.conv1 = nn.Conv2d(
            in_channels=3,        # 输入：3通道（RGB彩色图）
            out_channels=64,      # 输出：64通道特征图
            kernel_size=7,        # 7×7大卷积核：快速压缩尺寸
            stride=2,             # 步长2：图片尺寸缩小一半（比如224×224→112×112）
            padding=3,            # 边缘填充：保证卷积后尺寸正确
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1  # 3×3池化核：尺寸再缩小一半（112×112→56×56）
        )
        
        # 2. 4组残差块（共8个BasicBlock，对应ResNet-18的“18层”中16个卷积层）
        # 每组残差块的参数：(块数, 输出通道数, 步长)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # 2个块，输出64通道，尺寸56×56（不变）
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2) # 2个块，输出128通道，尺寸28×28（缩小一半）
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2) # 2个块，输出256通道，尺寸14×14（缩小一半）
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2) # 2个块，输出512通道，尺寸7×7（缩小一半）
        
        # 3. 全局平均池化：把7×7×512的特征图，转成1×1×512的向量（每个通道取平均值）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 不管输入尺寸，输出都是(1,1)
        
        # 4. 全连接层：把512维向量，转成“类别数”维的输出（比如1000类，输出1000个数值）
        self.fc = nn.Linear(512, num_classes)
    
    # 辅助函数：批量创建残差块（避免重复写代码）
    def _make_layer(self, block, out_channels, blocks_num, stride=1):
        layers = []  # 用列表存所有残差块
        
        # 第一块残差块：可能需要步长stride（缩小尺寸）或适配通道，所以单独创建
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels  # 更新后续块的“输入通道数”（和当前输出通道一致）
        
        # 剩下的blocks_num-1块：步长固定为1（不改变尺寸），通道数已适配
        for _ in range(blocks_num - 1):
            layers.append(block(self.in_channels, out_channels, stride=1))
        
        # 用“序列容器”把所有块打包，返回一个“组”
        return nn.Sequential(*layers)
    
    # 定义整个模型的数据流动路径
    def forward(self, x):
        # 1. 初始处理：conv1 → bn1 → relu → maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 2. 4组残差块：逐层提取更复杂的特征
        x = self.layer1(x)  # 56×56×64 → 56×56×64
        x = self.layer2(x)  # 56×56×64 → 28×28×128
        x = self.layer3(x)  # 28×28×128 → 14×14×256
        x = self.layer4(x)  # 14×14×256 → 7×7×512
        
        # 3. 池化+全连接：输出类别
        x = self.avgpool(x)  # 7×7×512 → 1×1×512
        x = torch.flatten(x, 1)  # 把(1×1×512)展平成(512,)的向量（去掉空间维度）
        x = self.fc(x)  # 512 → num_classes（比如1000）
        
        return x
    
    # 第三步：验证模型—让ResNet-18跑起来
def test_resnet18():
    """测试ResNet-18模型"""
    print("开始测试ResNet-18模型...")

    # 创建模型实例
    model = ResNet18(num_classes=1000)
    model.eval()  # 评估模式

    # 模拟输入：1张3通道224x224的图片
    fake_image = torch.randn(1, 3, 224, 224)
    print(f"输入图片形状：{fake_image.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(fake_image)

    # 查看结果
    print(f"模型输出形状：{output.shape}")
    print(f"预测概率最高的类别索引：{torch.argmax(output, dim=1).item()}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params:,} 个参数")

    return model


# 额外功能：模型结构可视化
def print_model_summary(model, input_size=(3, 224, 224)):
    """打印模型结构摘要"""
    from torchsummary import summary

    print("\n" + "="*50)
    print("ResNet-18模型结构摘要")
    print("="*50)

    try:
        summary(model, input_size=input_size)
    except ImportError:
        print("⚠️ torchsummary未安装，跳过模型摘要")
        print("💡 安装命令：pip install torchsummary")


# 修改主函数部分
if __name__ == "__main__":
    # 1. 检测是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 测试模型
    model = test_resnet18()
    
    # 3. 【关键】将模型移动到 GPU
    model = model.to(device) 

    # 4. 打印模型结构
    # torchsummary 默认会使用 cuda，现在模型也在 cuda 上了，就不会报错了
    print_model_summary(model)

    print("\n✅ ResNet-18模型测试完成！")