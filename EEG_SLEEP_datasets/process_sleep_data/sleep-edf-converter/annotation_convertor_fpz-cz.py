# -*- coding: utf-8 -*-
"""
Modified for Sleep-EDF: 
1. Convert annotations to NPY
2. Extract EEG Fpz-Cz signals
3. Plot Signal + Hypnogram alignment
4. Robust validation (Input check + Output Image Integrity check)
5. Fixed MNE legacy warning (pick_channels -> pick)
6. Added feature: Skip saving NPY if file exists and content is identical.
"""
import os
import re
import sys
import json
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from optparse import OptionParser

"""
    功能:将数据集文件产生npy文件和信号图, 并将其保存到指定目录下,提取 EEG Fpz-Cz 通道,并将信号与标签(Hypnogram)绘制在同一张图的上下子图中，时间轴对齐
    运行:
    python3 annotation_convertor_fpz-cz.py /liuran/liuran/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette  /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/npy  /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_img
    注意：
    此文件和运行annotation_convertor.py文件的区别：
        都产生npy文件，只是在此文件中除了产生npy文件后，还选择了通道，直接运行此文件即可。
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

# 尝试导入 PIL 用于图像完整性校验
try:
    from PIL import Image
except ImportError:
    print("Error: 缺少 Pillow 库用于图像校验。请运行 `pip install pillow` 进行安装。")
    sys.exit(1)

def parse_timestamp(time_str):
    """解析时间字符串，兼容跨天和特殊格式"""
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

def validate_file(filepath):
    """校验文件是否存在且大小不为0"""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    if os.path.getsize(filepath) == 0:
        return False, f"File is empty (0 bytes): {filepath}"
    return True, "Valid"

def recognize_edf_or_edfx(path):
    out = None
    if len(util.search_file(path, '.edf')[1]) != 0:
        out = False # Sleep-EDFx
    elif len(util.search_file(path, '.rec')[1]) != 0:
        out = True  # Sleep-EDF (Old)
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
        print(f"Warning: Files mismatch! REC: {len(rec)}, HYP: {len(hyp)}")
    
    min_len = min(len(rec), len(hyp))
    return list(zip(rec[:min_len], hyp[:min_len]))
 
def read_header(file):
    out = None
    is_valid, msg = validate_file(file)
    if not is_valid:
        raise ValueError(msg)

    try:
        result = subprocess.run(['save2gdf', '-JSON', file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            try:
                raw_str = result.stdout.decode(errors='ignore').replace('\n','').replace('\r','').replace('\t','').replace('inf','0')
                out = json.loads(raw_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Header JSON Decode Error: {e}")
        else:
            err_msg = result.stderr.decode(errors='ignore').strip()
            raise ValueError(f"save2gdf failed. Msg: {err_msg}")
    except FileNotFoundError:
        raise FileNotFoundError('Command `save2gdf` not found in PATH.')
    return out

def get_start(file):
    header = read_header(file)
    if header is None or 'StartOfRecording' not in header:
        raise ValueError(f"Header missing 'StartOfRecording': {file}")
    str_start = header['StartOfRecording']
    out = parse_timestamp(str_start)
    if out is None:
        raise ValueError(f"Could not parse timestamp: '{str_start}' in {file}")
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

# =========================================================================
# 【修改点】save_npy 函数：增加内容一致性检查
# =========================================================================
def save_npy(path, ori_fname, array):
    """
    保存 .npy 文件。
    如果在目标路径下已存在该文件，且内容与当前计算的 array 完全一致，
    则跳过写入操作，避免重复 I/O。
    """
    if len(array) == 0:
        return False
    
    fname = os.path.join(path, os.path.splitext(ori_fname)[0] + '.npy')
    
    # 1. 检查文件是否存在
    if os.path.exists(fname):
        try:
            # 2. 读取旧文件
            existing_array = np.load(fname)
            # 3. 比较内容是否一致
            if np.array_equal(existing_array, array):
                print(f"Skipped NPY (Exists & Identical): {fname}")
                return True
        except Exception:
            # 如果读取旧文件出错，忽略错误，强制覆盖写入
            pass

    # 4. 写入新文件（如果不存在，或内容不一致，或读取旧文件失败）
    np.save(fname, array)
    print(f"Saved NPY: {fname}")
    return True

def plot_and_save_image(rec_path, npy_array, output_img_dir, filename):
    """
    绘制并保存图像，并在保存后立即校验图像的完整性。
    """
    # 1. 物理检查
    valid, msg = validate_file(rec_path)
    if not valid:
        raise ValueError(msg)

    # 2. 读取 EEG 信号 (MNE)
    try:
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose='ERROR')
    except Exception as e:
        raise ValueError(f"MNE read_raw_edf failed: {e}")

    # 3. 查找 Fpz-Cz 通道
    target_ch = None
    for ch in raw.ch_names:
        if "Fpz-Cz" in ch:
            target_ch = ch
            break
    
    if target_ch is None:
        raise ValueError(f"Channel 'Fpz-Cz' not found. Available: {raw.ch_names}")
    
    # 4. 提取数据
    try:
        # 使用 pick 替代 pick_channels
        raw_selection = raw.copy().pick([target_ch])
        signal_data, times = raw_selection[:] 
        signal_data = signal_data[0] * 1e6 # V -> uV
    except Exception as e:
        raise ValueError(f"Failed to extract channel data: {e}")
    
    # 5. 准备 Hypnogram 数据
    if npy_array.size == 0:
        raise ValueError("Hypnogram array is empty")

    onset_seconds = npy_array[:, 0].astype(float)
    stages = npy_array[:, 2]
    
    stage_map = {'W': 0, 'R': -1, '1': -2, '2': -3, '3': -4, '4': -4, 'M': 1, 'L': 1, '?': 1}
    hypno_y = [stage_map.get(s, 1) for s in stages]
    
    plot_onsets = np.append(onset_seconds, onset_seconds[-1] + 30)
    plot_stages = hypno_y + [hypno_y[-1]]

    # 6. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(times, signal_data, color='black', linewidth=0.3)
    ax1.set_title(f"Subject: {filename} | Channel: {target_ch}")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.step(plot_onsets, plot_stages, where='post', color='navy', linewidth=1.5)
    
    yticks = [1, 0, -1, -2, -3, -4]
    yticklabels = ['Mov', 'Wake', 'REM', 'N1', 'N2', 'N3/4']
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.set_ylabel("Sleep Stage")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax2.set_xlim(0, times[-1])
    ax2.set_ylim(-4.5, 1.5)
    
    # 7. 保存图片
    out_name = os.path.join(output_img_dir, os.path.splitext(filename)[0] + '.png')
    plt.tight_layout()
    plt.savefig(out_name, dpi=100)
    plt.close(fig) 

    # 8. 校验生成的图片
    try:
        with Image.open(out_name) as img:
            img.verify() 
    except Exception as e:
        if os.path.exists(out_name):
            os.remove(out_name)
        raise ValueError(f"IMAGE CORRUPTED: Generated image could not be opened. Deleted. Error: {e}")

def main():
    usage = 'usage: %prog [options] SOURCE NPY_DEST IMG_DEST';
    parser = OptionParser(usage = usage);
    (options, args) = parser.parse_args();
    
    if len(args) != 3:
        parser.print_usage(file = sys.stderr);
        sys.stderr.write("\tError: Need 3 arguments (Source, Npy_Dest, Img_Dest)\n");
        sys.exit(1);
    else:
        source, dest_npy, dest_img = args;
        
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
            
            print(f"Processing: {rec} ...", end=" ", flush=True)
            
            try:
                # 1. 基础验证
                v_rec, msg_rec = validate_file(full_rec_path)
                v_hyp, msg_hyp = validate_file(full_hyp_path)
                if not v_rec: raise ValueError(msg_rec)
                if not v_hyp: raise ValueError(msg_hyp)

                # 2. 解析 Hypnogram 数据
                start = get_start(full_rec_path);
                events = get_events(full_hyp_path);
                
                if not events:
                    raise ValueError("No events found in Hypnogram")

                array = relative(start, events);
                
                # 3. 保存 NPY (带一致性检查)
                save_npy(dest_npy, rec, array);
                
                # 4. 绘图 + 图像完整性校验
                plot_and_save_image(full_rec_path, array, dest_img, rec)
                
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
        print(f"Successfully processed: {success_count}")
        print(f"Failed/Skipped files:   {len(damaged_files)}")
        
        if damaged_files:
            print("\n!!! LIST OF FAILED / CORRUPT FILES !!!")
            print(f"{'File Name':<30} | {'Reason for Failure'}")
            print("-" * 70)
            for name, reason in damaged_files:
                clean_reason = reason.replace('\n', ' ')
                if len(clean_reason) > 50:
                    clean_reason = clean_reason[:47] + "..."
                print(f"{name:<30} | {clean_reason}")
        print("="*60)

if __name__ == '__main__':
    try:
        main();
    except KeyboardInterrupt:
        sys.exit(1);