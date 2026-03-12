# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def inspect_npy(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return

    print(f"--- 正在检查文件: {os.path.basename(file_path)} ---")
    
    # 1. 加载数据
    # 注意：因为之前的代码保存的是混合类型(int, datetime, str)，所以必须开启 allow_pickle
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 打印基础形状和类型
    print(f"【基本信息】")
    print(f"  Shape (形状): {data.shape}")
    print(f"  Dtype (类型): {data.dtype}")
    print(f"  总事件数: {len(data)}")
    print("-" * 30)

    if len(data) == 0:
        print("警告: 数据为空！")
        return

    # 3. 打印内容预览
    print("【内容预览 (前 5 行)】")
    print(f"{'相对秒数(sec)':<15} | {'绝对时间(timestamp)':<26} | {'睡眠阶段(label)'}")
    print("-" * 60)
    for row in data[:5]:
        # row[0]: 相对秒数, row[1]: datetime对象, row[2]: 标签
        print(f"{str(row[0]):<15} | {str(row[1]):<26} | {row[2]}")
    
    print("\n【内容预览 (后 5 行)】")
    for row in data[-5:]:
        print(f"{str(row[0]):<15} | {str(row[1]):<26} | {row[2]}")
    print("-" * 30)

    # 4. 逻辑对齐检查 (Sanity Check)
    seconds = data[:, 0]
    
    # 检查单调性
    is_monotonic = np.all(seconds[1:] >= seconds[:-1])
    print(f"【对齐与逻辑检查】")
    print(f"  时间单调递增: {'✅ 是' if is_monotonic else '❌ 否 (警告: 时间顺序错乱)'}")
    
    # 检查起始时间
    start_sec = seconds[0]
    print(f"  记录开始于第: {start_sec} 秒")
    if start_sec > 3600:
        print("  ⚠️ 警告: 第一个标签在记录开始 1 小时后才出现，请确认这是否符合预期？")
    elif start_sec < 0:
        print("  ❌ 错误: 出现负数时间，说明标签时间早于记录开始时间！")
    else:
        print("  ✅ 起始时间看起来正常。")

    # 检查总时长
    duration_sec = seconds[-1] - seconds[0]
    duration_hrs = duration_sec / 3600.0
    print(f"  标签覆盖时长: {duration_sec} 秒 ({duration_hrs:.2f} 小时)")
    
    # 5. 可视化 Hypnogram (这是验证对齐最直观的方法)
    plot_hypnogram(data, os.path.basename(file_path))

def plot_hypnogram(data, filename):
    """
    绘制睡眠结构图
    """
    times = data[:, 0].astype(float) # 相对秒数
    stages_raw = data[:, 2]          # 标签
    
    # 映射字典：将字符标签转换为数字以便绘图
    #通常标准是: W=0, N1=-1, N2=-2, N3/4=-3, R=1 (或者 R放在最上面)
    # 这里我们用简单的阶梯图: W=0, R=-1, 1=-2, 2=-3, 3/4=-4, M/L=-5
    stage_map = {
        'W': 0, 
        'R': -1, 
        '1': -2, 
        '2': -3, 
        '3': -4, 
        '4': -4, # 将3和4合并为深睡
        'M': 1,  # 运动/伪影
        'L': 1,  # 未知
        '?': 1
    }
    
    # 转换
    y_values = [stage_map.get(s, 1) for s in stages_raw]
    
    plt.figure(figsize=(15, 6))
    
    # 绘制阶梯图
    plt.step(times, y_values, where='post', color='navy', linewidth=1.5)
    
    # 设置Y轴标签
    yticks = [1, 0, -1, -2, -3, -4]
    yticklabels = ['Mov/Unk', 'Wake', 'REM', 'N1', 'N2', 'N3']
    plt.yticks(yticks, yticklabels)
    
    # 设置X轴
    plt.xlabel("Time (seconds from start)")
    plt.title(f"Hypnogram check: {filename}")
    plt.grid(True, alpha=0.3)
    
    # 标出 0 点
    plt.axvline(x=0, color='r', linestyle='--', label='Start of Recording (0s)')
    plt.legend()
    
    plt.show()

# --- 这里填入你想检查的 .npy 文件路径 ---
if __name__ == "__main__":
    # 示例: 
    target_file = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/npy/SC4002E0-PSG.npy" 
    
    # 为了方便测试，如果你不传参数，你可以手动修改上面的 target_file
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    inspect_npy(target_file)