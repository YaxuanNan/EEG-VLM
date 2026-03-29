# -*- coding: utf-8 -*-
import os
import json
import glob
import re
import pandas as pd

"""
    功能：
        直接评估成功率，没有任何规则，只根据模型输出的文本是否包含正确的阶段来判断是否成功。
"""

# ================= 路径配置 =================
IMAGE_DIR = "/mnt/inaisfs/workspace/EEG-VLM/data/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/test"
ANSWERS_FILE = '/mnt/inaisfs/workspace/EEG-VLM/data/liuran/EEG/EEG_SLEEP_datasets/eval_eeg_successful/json/eeg_answers_G1_320e2a1b32_cot_1.jsonl'#每个阶段要更改序号
OUTPUT_DIR = '/mnt/inaisfs/workspace/EEG-VLM/data/liuran/EEG/EEG_SLEEP_datasets/eval_eeg_successful/excel'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'eeg_sleep_comparison_eeg_answers_G1_320e2a1b32_cot_1.xlsx')#每个阶段要更改序号,序号以X_0表示，表示无规则识别原则

# ================= 1. 逻辑配置 =================

def natural_sort_key(filepath):
    """自然排序算法"""
    filename = os.path.basename(filepath)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def get_label_from_filename(filename):
    """从文件名解析真实阶段"""
    if "stageW" in filename:   return "Wake (W)"
    if "stage1" in filename:   return "N1"
    if "stage2" in filename:   return "N2"
    if "stage3" in filename:   return "N3"
    if "stageR" in filename:   return "REM"
    return "Unknown"

def extract_stage_from_text(text):
    """
    读取 JSONL 中的 text 字段并输出相应阶段
    规则：读取第一句话（以'.'结束），如果有','则只读取','前面的部分。然后在该部分中查找阶段关键词。
    """
    if not text:
        return "Unknown"
    
    # 1. 提取第一句话：以句号 '.' 进行分割，并取第一部分
    first_sentence = text.split('.')[0]
    
    # 2. 如果第一句话中包含逗号 ','，则只取逗号前面的部分
    target_substring = first_sentence.split(',')[0]
    
    # 3. 转换为大写进行统一匹配
    target_text = target_substring.upper()
    
    # --- 标准阶段匹配模式 ---
    patterns = {
        "Wake (W)": r'\bWAKE\b|\(W\)', 
        "N1": r'\bN1\b',
        "N2": r'\bN2\b',
        "N3": r'\bN3\b',
        "REM": r'\bREM\b'
    }
    
    for stage, pattern in patterns.items():
        if re.search(pattern, target_text):
            return stage
            
    return "Unknown"

# ================= 2. 主处理逻辑 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 步骤 A: 读取图片并按自然排序 ---
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    image_files.sort(key=natural_sort_key)
    
    # --- 步骤 B: 读取 JSONL 的 text 字段 ---
    prediction_results = {}
    print("="*20 + " 正在读取 JSONL 文件并输出 text 内容 " + "="*20)
    
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 确保 question_id 存在且为整数
                    q_id_raw = item.get("question_id")
                    if q_id_raw is None: continue
                    q_id = int(q_id_raw)
                    
                    # 提取 text 并在终端输出
                    raw_text = item.get("text", "")
                    
                    # 【新增】在终端中输出 text 标识中的内容
                    print(f"[Q_ID: {q_id}] 模型输出内容(text): {raw_text}")
                    
                    # 转为相应阶段
                    pred_stage = extract_stage_from_text(raw_text)
                    prediction_results[q_id] = pred_stage
                except Exception:
                    continue
    else:
        print(f"❌ 找不到文件: {ANSWERS_FILE}")

    # --- 步骤 C: 数据对齐与对比统计 ---
    final_data = []
    success_count = 0
    unknown_count = 0  # 记录 Unknown 的数量
    total_count = len(image_files)

    print("\n" + "="*20 + " 正在进行模型生成文本与真实标签对比 " + "="*20)
    
    for i, img_path in enumerate(image_files):
        # 假设 question_id 是从 1 开始的自增索引（与图片排序对应）
        q_id = i + 1 
        filename = os.path.basename(img_path)
        
        true_stage = get_label_from_filename(filename)
        model_stage = prediction_results.get(q_id, "Missing")
        
        # 统计 Unknown 数量
        if model_stage == "Unknown":
            unknown_count += 1

        # 比较等价性
        is_success = (true_stage == model_stage)
        if is_success:
            success_count += 1
            
        final_data.append({
            "question_id": q_id,
            "filename": filename,
            "true_stage": true_stage,
            "model_stage": model_stage,
            "is_correct": "Yes" if is_success else "No"
        })

        if q_id % 100 == 0 or q_id == total_count:
            print(f"进度: {q_id}/{total_count} | 当前准确匹配数: {success_count} | 当前 Unknown 数: {unknown_count}")

    # 计算成功率
    success_rate = success_count / total_count if total_count > 0 else 0
    success_rate_pct = f"{success_rate:.2%}"

    # --- 步骤 D: 导出最终 Excel ---
    if final_data:
        df = pd.DataFrame(final_data)
        
        # 添加最后一行汇总数据，将 Unknown 数量也写入
        summary_row = pd.DataFrame([{
            "question_id": "Summary",
            "filename": f"Total: {total_count}",
            "true_stage": f"Correct: {success_count}",
            "model_stage": f"Unknown: {unknown_count}",  
            "is_correct": f"Accuracy: {success_rate_pct}" 
        }])
        
        df_final = pd.concat([df, summary_row], ignore_index=True)
        
        try:
            df_final.to_excel(OUTPUT_FILE, index=False)
            print("\n" + "="*40)
            print(f"📊 统计汇总完成：")
            print(f"总比较样本数: {total_count}")
            print(f"准确匹配数量: {success_count}")
            print(f"无法识别 (Unknown) 数量: {unknown_count}")  
            print(f"准确率 (Accuracy): {success_rate_pct}")
            print(f"比对结果已保存至: {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Excel 保存失败: {e}")

if __name__ == "__main__":
    main()