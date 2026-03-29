# -*- coding: utf-8 -*-
import os
import shutil
import random
import sys
from PIL import Image

"""
功能：
1. 完整性检测：通过加载像素数据确保图片未损坏。
2. 自动补充：遇到损坏图片时，自动从同阶段剩余图片中寻找补位。
3. 详细报告：打印坏图名单、补充图名单，以及最终的分阶段详细统计。
"""

# ================= 配置区域 =================
SOURCE_IMG_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_split30s_img"
TARGET_ROOT_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event"

TEST_SAMPLES_PER_STAGE = 75
TRAIN_TARGET_COUNTS = {
    'W': 1175,  # Wake
    '1': 1186,  # N1
    '2': 758,   # N2
    '3': 836,   # N3
    'R': 1165   # REM
}
# ===========================================

def is_image_complete(filepath):
    """
    通过实际加载像素数据来检测图片是否完整。
    verify() 只能检查文件头，load() 才能检查数据体是否截断。
    """
    try:
        with Image.open(filepath) as img:
            img.verify()  # 初步验证
        with Image.open(filepath) as img:
            img.load()    # 强制解码像素，最严格的检查
        return True
    except Exception:
        return False

def get_stage_from_filename(filename):
    name, _ = os.path.splitext(filename)
    if '_stage' in name:
        return name.split('_stage')[-1]
    return 'Unknown'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def select_valid_samples(pool, count, stage_label):
    """
    从 pool 中筛选出 count 个健康的图片。
    返回: (选中的文件列表, 剩余的 pool, 损坏的文件列表, 补位的图片列表)
    """
    selected = []
    bad_files = []
    supplementary = []
    
    # 我们认为 pool 的前 count 个是“首选预定”，之后的都是“备选补位”
    initial_threshold = count
    
    idx = 0
    while len(selected) < count and idx < len(pool):
        filename = pool[idx]
        filepath = os.path.join(SOURCE_IMG_DIR, filename)
        
        if is_image_complete(filepath):
            selected.append(filename)
            # 如果是在 index 超过预定数量后才被选中的，说明它是补位选手
            if idx >= initial_threshold:
                supplementary.append(filename)
        else:
            bad_files.append(filename)
        idx += 1
    
    remaining_pool = pool[idx:]
    return selected, remaining_pool, bad_files, supplementary

def main():
    print(">>> 启动：完整性校验与定量分配程序...")
    
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"错误: 找不到源目录 {SOURCE_IMG_DIR}")
        return

    # 1. 读取并分组
    all_files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_by_stage = {}
    for f in all_files:
        stage = get_stage_from_filename(f)
        if stage not in files_by_stage:
            files_by_stage[stage] = []
        files_by_stage[stage].append(f)

    # 2. 创建目录
    train_root = os.path.join(TARGET_ROOT_DIR, 'train')
    test_root = os.path.join(TARGET_ROOT_DIR, 'test')
    ensure_dir(train_root)
    ensure_dir(test_root)

    #用于存储最终分阶段的统计数据
    final_stats = {
        'train1': {},
        'test': {}
    }

    # 3. 按阶段处理 (W, 1, 2, 3, R)
    sorted_stages = sorted(TRAIN_TARGET_COUNTS.keys()) # 保证输出顺序一致
    
    for stage in sorted_stages:
        target_train = TRAIN_TARGET_COUNTS[stage]
        pool = files_by_stage.get(stage, [])
        
        if not pool:
            print(f"⚠️  阶段 {stage} 未发现任何源文件，跳过。")
            final_stats['train1'][stage] = 0
            final_stats['test'][stage] = 0
            continue

        # 固定随机种子确保过程可追溯
        random.seed(42)
        random.shuffle(pool)

        print(f"\n--- 正在处理阶段 [{stage}] ---")

        # --- 第一步：筛选 Test 集 ---
        test_selected, pool, test_bad, test_supp = select_valid_samples(pool, TEST_SAMPLES_PER_STAGE, stage)
        
        # --- 第二步：筛选 Train1 集 ---
        train_selected, _, train_bad, train_supp = select_valid_samples(pool, target_train, stage)

        # --- 记录统计数据 ---
        final_stats['test'][stage] = len(test_selected)
        final_stats['train1'][stage] = len(train_selected)

        # --- 日志打印 (坏图与补位) ---
        all_bad = test_bad + train_bad
        all_supp = test_supp + train_supp
        
        if all_bad:
            print(f"  ❌ 发现坏图 ({len(all_bad)}张): {', '.join(all_bad)}")
            if all_supp:
                print(f"  ✅ 自动补位 ({len(all_supp)}张): {', '.join(all_supp)}")
            else:
                print("  ⚠️  警告：备选池已耗尽，无法补齐缺口！")

        # --- 物理复制 ---
        def copy_files(files, target_dir):
            for f in files:
                shutil.copy(os.path.join(SOURCE_IMG_DIR, f), os.path.join(target_dir, f))
            return len(files)

        copy_files(test_selected, test_root)
        copy_files(train_selected, train_root)

    # 4. 最终详细统计报告
    print("\n" + "="*60)
    print(f"{'最终详细统计报告':^60}")
    print("="*60)
    
    # 打印 训练集 Train1 统计
    print(f"\n【训练集 (train1)】 目标目录: {train_root}")
    print(f"{'阶段':<10} | {'实际数量':<10} | {'目标数量':<10} | {'状态':<10}")
    print("-" * 50)
    train_total = 0
    for stage in sorted_stages:
        count = final_stats['train1'].get(stage, 0)
        target = TRAIN_TARGET_COUNTS[stage]
        status = "✅ 达标" if count == target else "⚠️ 不足"
        print(f" {stage:<10} | {count:<10} | {target:<10} | {status}")
        train_total += count
    print("-" * 50)
    print(f" 训练集总计 : {train_total} 张")

    # 打印 验证集 Test 统计
    print(f"\n【验证集 (test)】   目标目录: {test_root}")
    print(f"{'阶段':<10} | {'实际数量':<10} | {'目标数量':<10} | {'状态':<10}")
    print("-" * 50)
    test_total = 0
    for stage in sorted_stages:
        count = final_stats['test'].get(stage, 0)
        target = TEST_SAMPLES_PER_STAGE
        status = "✅ 达标" if count == target else "⚠️ 不足"
        print(f" {stage:<10} | {count:<10} | {target:<10} | {status}")
        test_total += count
    print("-" * 50)
    print(f" 验证集总计 : {test_total} 张")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()