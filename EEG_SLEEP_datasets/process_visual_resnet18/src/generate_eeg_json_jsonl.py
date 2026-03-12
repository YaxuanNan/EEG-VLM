# -*- coding: utf-8 -*-
import os
import json
import glob
import re
import random
import shutil  # 新增：用于复制文件
from collections import Counter

# ================= 路径配置 =================
# 1. 训练集图片路径
TRAIN_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/train"
# 2. 原始测试集图片路径 (源路径)
SOURCE_TEST_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/test"
# 3. 评测集图片路径 (目标路径，针对 test1 生成 jsonl)
EVAL_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/test1"
# 4. 输出目录
OUTPUT_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/eeg_vlm_sleep_data/eeg_json_train_eval"

# ================= 1. 逻辑配置 (解析与描述) =================
def get_label(filename):
    """根据文件名解析睡眠阶段标签"""
    if "stageW" in filename:   return "Wake (W)"
    if "stage1" in filename:   return "N1"
    if "stage2" in filename:   return "N2"
    if "stage3" in filename:   return "N3"
    if "stageR" in filename:   return "REM"
    return "Unknown"

def get_detailed_description(label):
    """详细特征描述库"""
    descriptions = {
        "Wake (W)": (
            "The image displays an EEG signal characteristic of the Wake (W) stage. "
            "It features prominent Alpha rhythm (8-13 Hz) activity, particularly in the posterior regions, "
            "mixed with low-voltage, high-frequency Beta activity (>13 Hz). "
            "Eye blink artifacts or muscle movements may also be visible."
        ),
        "N1": (
            "The image shows an EEG recording during the N1 sleep stage. "
            "The features include low-voltage, mixed-frequency Theta waves (4-7 Hz). "
            "The prominent Alpha rhythm seen in wakefulness is diminished or absent. "
            "Vertex sharp waves may be present, indicating the transition from wakefulness to sleep."
        ),
        "N2": (
            "This is an EEG signal corresponding to the N2 sleep stage. "
            "The image is strictly characterized by the presence of distinct sleep spindles "
            "(bursts of oscillatory brain activity at 12-14 Hz) and K-complexes "
            "(high-amplitude biphasic waves). These features appear on a background of low-voltage, "
            "mixed-frequency activity."
        ),
        "N3": (
            "The image depicts the N3 stage, also known as Slow Wave Sleep (SWS). "
            "It is strictly defined by the dominance of high-amplitude (>75 µV), "
            "slow-frequency Delta waves (0.5-2 Hz) which cover more than 20% of the epoch. "
            "The signal appears synchronized and high-voltage."
        ),
        "REM": (
            "The image illustrates the REM (Rapid Eye Movement) sleep stage. "
            "The EEG features low-voltage, mixed-frequency activity that resembles the waking state "
            "(desynchronized EEG). Distinctive 'sawtooth' waves may be observed. "
            "This stage is associated with rapid eye movements and muscle atonia."
        ),
        "Unknown": "The features of this EEG signal are ambiguous and do not clearly fit a specific sleep stage classification."
    }
    return descriptions.get(label, descriptions["Unknown"])

def natural_sort_key(filepath):
    """操作系统级“自然排序”算法"""
    filename = os.path.basename(filepath)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# ================= 2. 生成 eeg_eval.jsonl 逻辑 =================
def generate_eval_jsonl():
    """
    为 test1 目录准备数据并生成 eeg_eval.jsonl 文件 (JSONL格式)
    """
    print(f"\n{'='*20} 步骤 2: 开始处理评测集 (test -> test1 -> JSONL) {'='*20}")
    
    # 1. 检查并创建 test1 目录
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
        print(f"📁 目录不存在，已自动创建: {EVAL_DIR}")
    
    # 2. 从原始 test 目录复制并重命名图片到 test1 目录
    source_images = glob.glob(os.path.join(SOURCE_TEST_DIR, "*.png"))
    if not source_images:
        print(f"❌ 错误: 源测试集目录 {SOURCE_TEST_DIR} 中没有图片！请检查路径。")
        return

    print(f"⏳ 正在将图片从 test 复制并去标签重命名到 test1...")
    for img_path in source_images:
        filename = os.path.basename(img_path)
        
        # 使用正则表达式去掉 "_stage" 及其后面的所有字母数字
        # 例如: "subject1_epoch05_stageW.png" -> "subject1_epoch05.png"
        new_filename = re.sub(r'_stage\w+', '', filename)
        dest_path = os.path.join(EVAL_DIR, new_filename)
        
        # 复制文件
        shutil.copy2(img_path, dest_path)
        
    print(f"✅ 图片复制与重命名完成！")

    # 3. 读取 test1 中的新图片并自然排序
    eval_images = glob.glob(os.path.join(EVAL_DIR, "*.png"))
    sorted_images = sorted(eval_images, key=natural_sort_key)
    num_images = len(sorted_images)
    print(f"🔍 验证读取：在 {EVAL_DIR} 中共找到 {num_images} 张无标签图片。")

    output_path = os.path.join(OUTPUT_DIR, "eeg_eval.jsonl")
    
    # 评测集问题模板
    questions = [
        "Which sleep stage does this EEG segment represent?",
        "Can you identify the sleep stage shown in this image?",
        "Describe the sleep stage and the main features of this EEG signal.",
        "What is the current sleep stage according to this EEG recording?",
        "Based on the rhythmic activity in this image, which sleep stage is the subject in?"
    ]

    # 4. 写入 JSONL
    print(f"⏳ 开始生成 JSONL 文件...")
    with open(output_path, "w", encoding='utf-8') as f:
        for idx, img_path in enumerate(sorted_images):
            filename = os.path.basename(img_path)
            
            line_data = {
                "question_id": idx + 1,
                "image": filename,
                "text": random.choice(questions),
                "category": "detail"
            }
            f.write(json.dumps(line_data, ensure_ascii=False) + "\n")

            if (idx + 1) % 5000 == 0 or (idx + 1) == num_images:
                print(f"[Eval] 已处理进度: {idx+1}/{num_images}...")

    print(f"✅ eeg_eval.jsonl 文件已生成！保存路径: {output_path}")

# ================= 3. 训练集主逻辑 =================
def generate_and_print():
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 任务 1: 仅处理训练集 (eeg_ft.json) ---
    print(f"\n{'='*20} 步骤 1: 开始处理训练集 ({TRAIN_DIR}) {'='*20}\n")
    
    raw_images = glob.glob(os.path.join(TRAIN_DIR, "*.png"))
    if not raw_images:
        print(f"⚠️ 警告: 在 {TRAIN_DIR} 未找到图片文件！")
    else:
        sorted_images = sorted(raw_images, key=natural_sort_key)
        data_list = []
        train_stats = Counter()

        for idx, img_path in enumerate(sorted_images):
            filename = os.path.basename(img_path)
            label = get_label(filename)
            description = get_detailed_description(label)
            
            train_stats[label] += 1

            entry = {
                "id": f"eeg_ft_{idx}",
                "image": [filename, filename], 
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nDescribe the EEG features shown in this image strictly."
                    },
                    {
                        "from": "gpt",
                        "value": description 
                    }
                ]
            }
            data_list.append(entry)

            if (idx + 1) % 5000 == 0 or (idx + 1) == len(sorted_images):
                print(f"[训练集] 已处理: {idx+1}/{len(sorted_images)} 张...")

        save_path = os.path.join(OUTPUT_DIR, "eeg_ft.json")
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
            
        print(f"✅ eeg_ft.json 生成完成！路径: {save_path}")

        # 打印统计
        print(f"\n{'='*20} 训练集统计汇总 {'='*20}")
        stage_order = ["Wake (W)", "N1", "N2", "N3", "REM"]
        for stage in stage_order:
            print(f"✅ {stage}: {train_stats[stage]} 条数据")
        print(f"✅ 训练集总计: {len(data_list)} 条数据")

    # --- 任务 2: 执行评测集逻辑 (test -> test1 -> eeg_eval.jsonl) ---
    generate_eval_jsonl()

if __name__ == "__main__":
    generate_and_print()