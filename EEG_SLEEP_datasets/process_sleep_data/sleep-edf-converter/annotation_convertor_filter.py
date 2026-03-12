# -*- coding: utf-8 -*-
"""
Modified for Sleep-EDF: 
1. Generate NPY and Original Full-Night Plots (User Path).
2. Generate Filtered (0.5-35Hz), Full-Night Plots (Hardcoded Path).
3. Robust Validation: Ensure input data is valid and output images are not corrupted.
4. Smart Skip: Skip processing if valid outputs already exist.
"""
import os
import re
import sys
import json
import numpy as np
import subprocess
import matplotlib
# 强制使用 Agg 后端，防止内存溢出或无图形界面报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from optparse import OptionParser

"""
    功能:将数据集文件产生npy文件和信号图, 并将其保存到指定目录下,提取 EEG Fpz-Cz 通道,并将信号与标签(Hypnogram)绘制在同一张图的上下子图中，时间轴对齐
    运行:
    python3 annotation_convertor_filter.py /liuran/liuran/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette  /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/npy  /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_img
    注：
        此文件和运行annotation_convertor.py文件的区别：
            都产生npy文件，只是在此文件中除了产生npy文件后，还选择了通道并进行了滤波，直接运行此文件即可。
        新增功能：
            程序在目标文件夹中发现了以前生成的旧文件，但在检查旧文件时遇到了读取错误，于是程序自动判定旧文件“损坏/不可用”，并果断进行了重新计算和覆盖生成。
"""

# 引入自定义工具包
try:
    import util
except ImportError:
    print("Error: 缺少 'util.py' 文件，请确保它在当前目录下。")
    sys.exit(1)

# 尝试导入 mne
try:
    import mne
except ImportError:
    print("Error: 缺少 mne 库。请运行 `pip install mne` 进行安装。")
    sys.exit(1)

# 【新增】引入 PIL 用于校验生成的图片是否损坏
try:
    from PIL import Image
except ImportError:
    print("Error: 缺少 Pillow 库。请运行 `pip install pillow` 进行安装。")
    sys.exit(1)

# --- 【配置】用户指定的滤波后图片输出绝对路径 ---
FILTER_IMG_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_filter_img"

def validate_file(filepath):
    """【新增】物理校验文件是否存在且大小不为0"""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    if os.path.getsize(filepath) == 0:
        return False, f"File is empty (0 bytes): {filepath}"
    return True, "Valid"

def verify_image_integrity(image_path):
    """【新增】校验生成的图片是否损坏"""
    if not os.path.exists(image_path):
        return False, "Image file was not created."
    try:
        with Image.open(image_path) as img:
            img.verify() # 尝试读取文件头和结构
        return True, "Valid"
    except Exception as e:
        return False, f"Image corruption detected: {e}"

def check_skip_condition(filename_base, dest_npy, dest_img):
    """
    【新增功能】检查目标路径下是否已有有效文件。
    返回: (是否跳过: bool, 原因/错误信息: str)
    """
    target_npy = os.path.join(dest_npy, filename_base + '.npy')
    target_img = os.path.join(dest_img, filename_base + '.png')

    # 1. 检查文件是否存在
    if not os.path.exists(target_npy):
        return False, "NPY file missing"
    if not os.path.exists(target_img):
        return False, "Image file missing"

    # 2. 检查 NPY 是否完整
    try:
        data = np.load(target_npy)
        if data.size == 0:
            return False, "Existing NPY is empty"
    except Exception as e:
        return False, f"Existing NPY corrupted: {e}"

    # 3. 检查图片是否完整
    valid_img, msg = verify_image_integrity(target_img)
    if not valid_img:
        return False, f"Existing Image corrupted: {msg}"

    return True, "Files exist and are valid"

def parse_timestamp(time_str):
    """解析时间字符串"""
    dt_out = None
    add_delta = timedelta(0)
    
    if ' 24:' in time_str:
        time_str = time_str.replace(' 24:', ' 00:')
        add_delta += timedelta(days=1)
        
    if re.search(r':60(?=\.|$)', time_str):
        time_str = re.sub(r':60(?=\.|$)', ':00', time_str, count=1)
        add_delta += timedelta(minutes=1)

    time_formats = [
        '%Y-%m-%d %H:%M:%S.%f', 
        '%Y-%m-%d %H:%M:%S',    
        '%Y-%b-%d %H:%M:%S',    
        '%Y-%b-%d %H:%M:%S.%f'  
    ]
    
    for fmt in time_formats:
        try:
            dt_out = datetime.strptime(time_str, fmt)
            break
        except ValueError:
            continue
            
    if dt_out is not None:
        dt_out += add_delta
        
    return dt_out

def recognize_edf_or_edfx(path):
    out = None
    if len(util.search_file(path, '.edf')[1]) != 0:
        out = False
    elif len(util.search_file(path, '.rec')[1]) != 0:
        out = True
    else:
        raise ValueError('Your input directory seems not a Sleep-EDF or Sleep-EDFx directory.')
    return out

def get_list(is_sleep_edf, path):
    out = None
    if is_sleep_edf:
        rec = util.search_file(path, '0.rec$')[1]
        rec.sort()
        hyp = util.search_file(path, '0.hyp$')[1]
        hyp.sort()
    else:
        rec = util.search_file(path, '-PSG.edf$')[1]
        rec.sort()
        hyp = util.search_file(path, '-Hypnogram.edf$')[1]
        hyp.sort()
    
    if len(rec) != len(hyp):
        print(f"Warning: File count mismatch! REC: {len(rec)}, HYP: {len(hyp)}")
    
    # 取交集长度防止索引越界
    min_len = min(len(rec), len(hyp))
    return list(zip(rec[:min_len], hyp[:min_len]))
 
def read_header(file):
    out = None
    # 增加文件校验
    valid, msg = validate_file(file)
    if not valid:
        raise ValueError(msg)

    try:
        result = subprocess.run(['save2gdf', '-JSON', file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            try:
                # 兼容处理输出中的特殊字符
                res_str = result.stdout.decode(errors='ignore').replace('\n','').replace('\r','').replace('\t','').replace('inf','0')
                out = json.loads(res_str)
            except json.JSONDecodeError:
                raise ValueError("Failed to decode JSON header")
        else:
            raise ValueError('save2gdf execution failed. Check if file is valid EDF.')
    except FileNotFoundError:
        raise FileNotFoundError('The required command `save2gdf` seems not appearing in your environment variables.')
    return out

def get_start(file):
    header = read_header(file)
    if header is None or 'StartOfRecording' not in header:
        raise ValueError(f"Failed to read header or 'StartOfRecording' missing in file: {file}")
    str_start = header['StartOfRecording']
    out = parse_timestamp(str_start)
    if out is None:
        raise ValueError(f"Could not parse StartOfRecording timestamp: '{str_start}' in file {file}")
    return out

def time_delta(label_timestamp, start):
    if not isinstance(label_timestamp, datetime) or not isinstance(start, datetime):
        return 0
    dt = label_timestamp - start
    out = int(dt.total_seconds()) 
    return out

def get_events(file):
    out = []
    header = read_header(file)
    if 'EVENT' not in header:
        return []
        
    events = header['EVENT']
    for event in events:
        time_obj = parse_timestamp(event['TimeStamp'])
        if time_obj is None:
            continue
        description = event.get('Description', '')
        if re.search('1$', description): stage = '1'
        elif re.search('2$', description): stage = '2'
        elif re.search('3$', description): stage = '3'
        elif re.search('4$', description): stage = '4'
        elif re.search('W$', description): stage = 'W'
        elif re.search('R$', description): stage = 'R'
        elif re.search('M$', description): stage = 'M'
        else: stage = 'L'
        out.append((time_obj, stage))
    return out

def relative(start, events):
    out = []
    for event in events:
        if event[0] is None: continue
        out.append([time_delta(event[0], start), event[0], event[1]])
    return np.array(out)

def save_npy(path, ori_fname, array):
    if len(array) == 0:
        print(f"Warning: Empty array for {ori_fname}, skipping NPY save.")
        return
    fname = os.path.join(path, os.path.splitext(ori_fname)[0] + '.npy')
    np.save(fname, array)
    print(f"Saved NPY: {fname}")

def plot_and_save_image(rec_path, npy_array, output_img_dir, filename):
    """
    [原始功能] 读取 EEG 信号 (无滤波)，结合标签绘制波形图并保存到用户指定目录
    """
    # 1. 物理校验
    valid, msg = validate_file(rec_path)
    if not valid:
        raise ValueError(msg)

    # 2. 读取数据
    try:
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose='ERROR')
    except Exception as e:
        raise ValueError(f"MNE read failed: {e}")

    # 3. 查找通道
    target_ch = None
    for ch in raw.ch_names:
        if "Fpz-Cz" in ch:
            target_ch = ch
            break
    
    if target_ch is None:
        raise ValueError(f"Channel 'Fpz-Cz' not found. Available: {raw.ch_names}")
    
    # 4. 提取数据
    raw_selection = raw.copy().pick([target_ch])
    signal_data, times = raw_selection[:]
    signal_data = signal_data[0] * 1e6 # 转换为 uV
    
    # 5. 准备标签
    if npy_array.size == 0:
        raise ValueError("Hypnogram array is empty")

    onset_seconds = npy_array[:, 0].astype(float)
    stages = npy_array[:, 2]
    
    stage_map = {'W': 0, 'R': -1, '1': -2, '2': -3, '3': -4, '4': -4, 'M': 1, 'L': 1, '?': 1}
    hypno_y = [stage_map.get(s, 1) for s in stages]
    
    # 6. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(times, signal_data, color='k', linewidth=0.5)
    ax1.set_title(f"EEG Signal ({target_ch}) - {filename}")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.grid(True, alpha=0.3)
    
    ax2.step(onset_seconds, hypno_y, where='post', color='navy')
    yticks = [1, 0, -1, -2, -3, -4]
    yticklabels = ['Mov', 'Wake', 'REM', 'N1', 'N2', 'N3/4']
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.set_ylabel("Sleep Stage")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, times[-1])
    
    out_name = os.path.join(output_img_dir, os.path.splitext(filename)[0] + '.png')
    plt.tight_layout()
    plt.savefig(out_name, dpi=100)
    plt.close(fig) 
    
    # 7. 校验生成的图片
    img_valid, img_msg = verify_image_integrity(out_name)
    if not img_valid:
        if os.path.exists(out_name): os.remove(out_name)
        raise ValueError(f"Generated raw image is corrupt: {img_msg}")
        
    print(f"Saved Raw Image: {out_name}")

def save_filtered_full_night_image(rec_path, npy_array, filename):
    """
    【核心功能】
    1. 滤波: Fpz-Cz, 0.5-35Hz, 1阶巴特沃斯
    2. 严格校验: 确保滤波后的数据非 NaN，确保生成的图片可打开
    3. 保存: 到指定绝对路径
    """
    # 1. 物理校验
    valid, msg = validate_file(rec_path)
    if not valid:
        print(f" [Filter Skip] {msg}")
        return

    # 2. 读取数据
    try:
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose='ERROR')
    except Exception as e:
        print(f" [Filter Skip] Error reading EDF: {e}")
        return

    # 3. 查找通道
    target_ch = None
    for ch in raw.ch_names:
        if "Fpz-Cz" in ch:
            target_ch = ch
            break
            
    if target_ch is None:
        print(f" [Filter Skip] Channel 'Fpz-Cz' not found in {filename}")
        return

    # 4. 应用滤波器
    print(f"  Filtering {filename} (0.5-35Hz)...")
    try:
        raw.filter(l_freq=0.5, h_freq=35.0, picks=[target_ch], method='iir', 
                   iir_params={'order': 1, 'ftype': 'butter'}, verbose=False)
    except Exception as e:
        print(f" [Filter Skip] Filtering failed: {e}")
        return

    # 5. 获取数据并校验数值有效性
    raw_selection = raw.copy().pick([target_ch])
    signal_data, times = raw_selection[:]
    signal_data = signal_data[0] * 1e6 # 转换单位为 uV

    # 【校验】检查是否存在 NaN 或 Inf
    if np.isnan(signal_data).any() or np.isinf(signal_data).any():
        print(f" [Filter Skip] Filtered data contains NaN or Inf values. Data invalid.")
        return

    # 6. 准备标签
    if npy_array.size == 0:
        print(" [Filter Skip] Hypnogram is empty.")
        return

    onset_seconds = npy_array[:, 0].astype(float)
    stages = npy_array[:, 2]
    
    stage_map = {'W': 0, 'R': -1, '1': -2, '2': -3, '3': -4, '4': -4, 'M': 1, 'L': 1, '?': 1}
    hypno_y = [stage_map.get(s, 1) for s in stages]

    # 7. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(times, signal_data, color='k', linewidth=0.5)
    ax1.set_title(f"[Filtered 0.5-35Hz] EEG Signal ({target_ch}) - {filename}")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.grid(True, alpha=0.3)
    
    ax2.step(onset_seconds, hypno_y, where='post', color='navy')
    yticks = [1, 0, -1, -2, -3, -4]
    yticklabels = ['Mov', 'Wake', 'REM', 'N1', 'N2', 'N3/4']
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.set_ylabel("Sleep Stage")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, times[-1])
    
    # 8. 保存与校验
    if not os.path.exists(FILTER_IMG_DIR):
        try:
            os.makedirs(FILTER_IMG_DIR)
        except Exception as e:
            print(f" [Filter Skip] Could not create output dir: {e}")
            return

    out_name = os.path.join(FILTER_IMG_DIR, os.path.splitext(filename)[0] + '_filtered.png')
    plt.tight_layout()
    plt.savefig(out_name, dpi=100)
    plt.close(fig)

    # 【校验】图片完整性检查
    img_valid, img_msg = verify_image_integrity(out_name)
    if not img_valid:
        print(f" [Filter Skip] Generated image corrupted: {img_msg}")
        if os.path.exists(out_name):
            os.remove(out_name) # 删除坏文件
    else:
        print(f"  Saved Filtered Full-Night Image: {out_name}")

def main():
    usage = 'usage: %prog [options] SOURCE NPY_DEST IMG_DEST';
    parser = OptionParser(usage = usage);
    (options, args) = parser.parse_args();
    
    if len(args) != 3:
        parser.print_usage(file = sys.stderr);
        sys.stderr.write("\tNeed 3 arguments: Source, Npy_Dest, Image_Dest\n\n");
        sys.exit(1);
    else:
        source, dest_npy, dest_img = args;
        
        # 检查并创建文件夹
        if not os.path.isdir(source): 
            print(f"Error: Source directory '{source}' does not exist.")
            sys.exit(1)
        
        if not os.path.exists(dest_npy):
            os.makedirs(dest_npy)
            
        if not os.path.exists(dest_img):
            os.makedirs(dest_img)

        try:
            data_entries = get_list(recognize_edf_or_edfx(source), source);
        except Exception as e:
            print(f"Error initializing file list: {e}")
            sys.exit(1)
        
        print(f"Found {len(data_entries)} pairs. Starting processing...")
        
        damaged_files = []
        success_count = 0

        for entry in data_entries:
            rec, hyp = entry;
            full_rec_path = os.path.join(source, rec)
            full_hyp_path = os.path.join(source, hyp)
            
            # --- 【关键修改】检查是否跳过 ---
            filename_base = os.path.splitext(rec)[0]
            should_skip, skip_reason = check_skip_condition(filename_base, dest_npy, dest_img)
            
            if should_skip:
                print(f"Skipping {rec}: {skip_reason}")
                success_count += 1
                continue
            else:
                # 如果不能跳过，说明文件不存在或者损坏，打印报错提示，然后继续执行生成逻辑
                if "missing" not in skip_reason: # 只有文件损坏才报错，文件缺失属于正常初次运行
                    print(f"[WARNING/ERROR] Found corrupt output files for {rec}: {skip_reason}. Regenerating...")
            
            print(f"Processing: {rec} ...", end=" ", flush=True)
            
            try:
                # 1. 处理标签
                start = get_start(full_rec_path);
                events = get_events(full_hyp_path);
                if not events:
                     raise ValueError("No events found in Hypnogram.")
                array = relative(start, events);
                
                # 2. 保存 .npy
                save_npy(dest_npy, rec, array);
                
                # 3. 绘制原始信号波形 (带校验)
                plot_and_save_image(full_rec_path, array, dest_img, rec)

                # 4. 绘制滤波后的整晚波形 (带数值校验和图片校验)
                save_filtered_full_night_image(full_rec_path, array, rec)
                
                print("[OK]")
                success_count += 1
                
            except Exception as e:
                print("[FAILED]")
                print(f"  -> Error: {e}")
                damaged_files.append((rec, str(e)))
                continue
        
        print("\n" + "="*60)
        print("PROCESSING REPORT")
        print("="*60)
        print(f"Successfully processed (or skipped valid): {success_count}")
        print(f"Failed/Skipped files:   {len(damaged_files)}")
        
        if damaged_files:
            print("\n=== Summary of Failed Files ===")
            for name, reason in damaged_files:
                print(f"{name}: {reason}")

if __name__ == '__main__':
    try:
        main();
    except KeyboardInterrupt:
        sys.exit(1);