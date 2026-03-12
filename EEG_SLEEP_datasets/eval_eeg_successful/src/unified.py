# -*- coding: utf-8 -*-
import os
import json
import glob
import re
import pandas as pd

"""
    功能：
        按照规则(规则描述在unified_eeg_processor.py文件中)评估成功率，eval_eeg_vlm_answers.py文件则直接评估成功率
    规则如下：
    1. 数据预处理：读取 unknown.jsonl，按 question_id 排序。
    2. 数据分流：
       - 不含 "indicating" 的数据：存入 unincluding_indicating.jsonl。若首句含 "waking state"，判定为 Wake (W)。
       - 包含 "indicating" 的数据：进入特征提取与投票逻辑。
    3. 特征提取：使用正则提取整段文本中所有 "from...to..." 片段（遇标点符号截断）。
    4. 倾向性决策（投票法）：
       - 取消立即熔断，遍历整段话中所有片段。
       - 命中 N1 特征（sleep stage/sleep）或 Wake 特征（waking/wakefulness）。
       - 最终对比总票数：多者胜出；平票且 >0 则优先判定为 N1。
    5. 统一输出：
       - unincluding_indicating.jsonl：存放不含 "indicating" 的原始数据。
       - read_from_to.jsonl：存放截取的特征文本。
       - sleep_stages.jsonl：记录所有 question_id 最终判定的状态映射表。
"""

# ================= 路径配置 =================
IMAGE_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/test"
ANSWERS_FILE = '/liuran/liuran/EEG/EEG_SLEEP_datasets/eval_eeg_successful/json/eeg_answers_3.jsonl' #每个训练阶段要更改序号
JSON_OUT_DIR = '/liuran/liuran/EEG/EEG_SLEEP_datasets/eval_eeg_successful/json'
EXCEL_OUT_DIR = '/liuran/liuran/EEG/EEG_SLEEP_datasets/eval_eeg_successful/excel'

# 输出文件定义,每个阶段要更改序号，序号以X_1表示，代表在规则下评估
OUTPUT_EXCEL = os.path.join(EXCEL_OUT_DIR, 'eeg_sleep_unified_comparison_3_1.xlsx')
UNKNOWN_JSONL = os.path.join(JSON_OUT_DIR, 'unknown_3_1.jsonl')
READ_FROM_TO_JSONL = os.path.join(JSON_OUT_DIR, 'read_from_to_3_1.jsonl')
SLEEP_STAGES_JSONL = os.path.join(JSON_OUT_DIR, 'sleep_stages_3_1.jsonl')

# ================= 1. 核心工具逻辑 =================

def natural_sort_key(filepath):
    """自然排序算法，确保 1.png < 2.png < 10.png"""
    filename = os.path.basename(filepath)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def get_label_from_filename(filename):
    """从文件名解析真实阶段标签"""
    if "stageW" in filename:   return "Wake (W)"
    if "stage1" in filename:   return "N1"
    if "stage2" in filename:   return "N2"
    if "stage3" in filename:   return "N3"
    if "stageR" in filename:   return "REM"
    return "Unknown"

def advanced_stage_decision(text):
    """
    整合投票决策与关键词提取的复合决策引擎
    """
    if not text: return "Unknown", "Empty Content"
    
    # --- 1. 基础关键词强制扫描 ---
    target_text = text.upper()
    patterns = {
        "Wake (W)": r'\bWAKE\b|\(W\)', 
        "N1": r'\bN1\b', "N2": r'\bN2\b', "N3": r'\bN3\b', "REM": r'\bREM\b'
    }
    for stage, pattern in patterns.items():
        if re.search(pattern, target_text):
            return stage, "Keyword Match"

    # --- 2. 倾向性投票逻辑 (针对复杂描述如 transition from...to...) ---
    matches = re.findall(r'(from\b.*?\bto\b[^,;:?!\(\)\[\]"\'，。；！？]*)', text, re.IGNORECASE)
    if matches:
        n1_score, wake_score = 0, 0
        for match in matches:
            parts = re.split(r'\bto\b', match, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                after_to = parts[1].lower()
                # N1 关键词特征
                if any(kw in after_to for kw in ["sleep stage", "sleep"]): n1_score += 1
                # Wake 关键词特征
                if any(kw in after_to for kw in ["waking state", "wakefulness state", "wakefulness"]): wake_score += 1
        
        if n1_score > wake_score: return "N1", "Voting (N1 Win)"
        elif wake_score > n1_score: return "Wake (W)", "Voting (Wake Win)"
        elif n1_score == wake_score and n1_score > 0: return "N1", "Voting (Tie-N1)"

    # --- 3. 首句保底匹配 ---
    first_sentence = text.split('.')[0].lower()
    if "waking state" in first_sentence:
        return "Wake (W)", "First Sentence Match"

    return "Unknown", "Unidentified"

# ================= 2. 主处理流程 =================

def main():
    # 确保输出目录存在
    for d in [JSON_OUT_DIR, EXCEL_OUT_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    # A. 准备图片序列（真值）
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    image_files.sort(key=natural_sort_key)
    
    # B. 加载模型输出并实时在终端打印回显
    prediction_data = {}
    print("="*20 + " 正在读取 JSONL 文件并输出内容回显 " + "="*20)
    
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q_id_raw = item.get("question_id")
                    if q_id_raw is None: continue
                    q_id = int(q_id_raw)
                    
                    raw_text = item.get("text", "")
                    # 实时回显功能
                    print(f"[Q_ID: {q_id}] 模型输出内容(text): {raw_text}")
                    
                    prediction_data[q_id] = item
                except Exception as e:
                    print(f"解析异常: {e}")
                    continue
    else:
        print(f"❌ 找不到输入文件: {ANSWERS_FILE}")
        return

    # C. 执行数据对齐、判定分流与对比统计
    final_data = []
    success_count = 0
    unknown_total = 0
    total_samples = len(image_files)

    print("\n" + "="*20 + " 正在执行分流分析与 Excel 统计 " + "="*20)

    # 打开分流输出文件流
    with open(UNKNOWN_JSONL, 'w', encoding='utf-8') as f_unk, \
         open(READ_FROM_TO_JSONL, 'w', encoding='utf-8') as f_rft, \
         open(SLEEP_STAGES_JSONL, 'w', encoding='utf-8') as f_ss:

        for i, img_path in enumerate(image_files):
            q_id = i + 1
            filename = os.path.basename(img_path)
            true_stage = get_label_from_filename(filename)
            
            # 获取预测结果，若不存在则记为 Missing
            item = prediction_data.get(q_id, {"text": ""})
            raw_text = item.get("text", "")
            
            # 执行复合逻辑决策
            model_stage, reason = advanced_stage_decision(raw_text)
            
            # --- 1. Unknown 提取分流 ---
            if model_stage == "Unknown":
                unknown_total += 1
                f_unk.write(json.dumps({"question_id": q_id, "text": raw_text}, ensure_ascii=False) + '\n')
            
            # --- 2. indicating 特征提取分流 ---
            if "indicating" in raw_text.lower():
                matches = re.findall(r'(from\b.*?\bto\b[^,;:?!\(\)\[\]"\'，。；！？]*)', raw_text, re.IGNORECASE)
                f_rft.write(json.dumps({"question_id": q_id, "extracted": matches}, ensure_ascii=False) + '\n')
            
            # --- 3. 所有状态映射表记录 ---
            f_ss.write(json.dumps({"question_id": q_id, "state": model_stage, "method": reason}, ensure_ascii=False) + '\n')

            # --- 4. 对比统计 ---
            is_correct = (true_stage == model_stage)
            if is_correct: success_count += 1
            
            final_data.append({
                "question_id": q_id,
                "filename": filename,
                "true_stage": true_stage,
                "model_stage": model_stage,
                "is_correct": "Yes" if is_correct else "No",
                "decision_reason": reason,
                "model_output": raw_text[:500]  # 限制长度
            })

    # D. 生成汇总统计与导出 Excel
    accuracy_pct = f"{(success_count / total_samples):.2%}" if total_samples > 0 else "0%"
    
    if final_data:
        df = pd.DataFrame(final_data)
        summary_row = pd.DataFrame([{
            "question_id": "Summary",
            "filename": f"Total: {total_samples}",
            "true_stage": f"Correct: {success_count}",
            "model_stage": f"Unknown: {unknown_total}",
            "is_correct": f"Accuracy: {accuracy_pct}",
            "decision_reason": "", "model_output": ""
        }])
        
        df_final = pd.concat([df, summary_row], ignore_index=True)
        
        try:
            df_final.to_excel(OUTPUT_EXCEL, index=False)
            print("\n" + "="*40)
            print(f"📊 统一分析汇总完成：")
            print(f"  ▶ 总处理样本数: {total_samples}")
            print(f"  ▶ 准确匹配数量: {success_count}")
            print(f"  ▶ 无法识别 (Unknown) 数量: {unknown_total}")
            print(f"  ▶ 整体准确率: {accuracy_pct}")
            print(f"📁 结果已导出：")
            print(f"  - 详细 Excel: {OUTPUT_EXCEL}")
            print(f"  - Unknown 记录: {UNKNOWN_JSONL}")
        except Exception as e:
            print(f"❌ Excel 保存失败: {e}")

if __name__ == "__main__":
    main()