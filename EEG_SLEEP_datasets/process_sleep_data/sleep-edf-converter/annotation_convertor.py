# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:25:22 2019

@author: Zhao Kuangshi
"""
import os;
import re;
import sys;
import util;
import json;
import numpy;
import subprocess;
from datetime import datetime, timedelta;
from optparse import OptionParser;

"""
    功能：
        将数据集文件产生npy文件（计算机秒读的二进制矩阵）, 并将其保存到指定目录下。
    修改：
    1. 增加错误统计列表，并在结束后打印所有损坏的文件名单
    2. 增加自动创建目标目录（npy文件夹）的功能
    python3 annotation_convertor.py /liuran/liuran/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/npy
    注意：
        此文件只是单纯产生npy,并没有选择通道，后面运行annotation_convertor_fpz-cz.py文件时与此文件功能重复，此文件运行可直接跳过。

"""

def parse_timestamp(time_str):
    """
    解析时间字符串，兼容 '24:xx:xx' (次日) 和 'xx:xx:60' (下一分) 的异常格式
    """
    dt_out = None
    add_delta = timedelta(0)
    
    # 1. 处理 24:xx:xx -> 00:xx:xx + 1 day
    if ' 24:' in time_str:
        time_str = time_str.replace(' 24:', ' 00:')
        add_delta += timedelta(days=1)
        
    # 2. 处理 xx:xx:60 -> xx:xx:00 + 1 minute
    if re.search(r':60(?=\.|$)', time_str):
        time_str = re.sub(r':60(?=\.|$)', ':00', time_str, count=1)
        add_delta += timedelta(minutes=1)

    # 3. 尝试多种格式解析
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
            
    # 4. 应用时间偏移
    if dt_out is not None:
        dt_out += add_delta
        
    return dt_out

def recognize_edf_or_edfx(path):
    # edf是True，edfx是False
    out = None;
    if len(util.search_file(path, '.edf')[1]) != 0:
        out = False;
    elif len(util.search_file(path, '.rec')[1]) != 0:
        out = True;
    else:
        raise ValueError('Your input directory seems not a Sleep-EDF or Sleep-EDFx directory.');
    return out;

def get_list(is_sleep_edf, path):
    # 根据所给的路径和数据集类型，得到打好包的文件列表
    if is_sleep_edf == True:
        rec = util.search_file(path, '0.rec$')[1];
        rec.sort();
        hyp = util.search_file(path, '0.hyp$')[1];
        hyp.sort();
    else:
        rec = util.search_file(path, '-PSG.edf$')[1];
        rec.sort();
        hyp = util.search_file(path, '-Hypnogram.edf$')[1];
        hyp.sort();
    
    if len(rec) != len(hyp):
        print(f"Warning: Number of recordings ({len(rec)}) and hypnograms ({len(hyp)}) do not match!")
        
    min_len = min(len(rec), len(hyp))
    out = list(zip(rec[:min_len], hyp[:min_len]));
    return out;

def validate_file(filepath):
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return False
    if os.path.getsize(filepath) == 0:
        print(f"[Error] File is empty (0 bytes): {filepath}")
        return False
    return True

def read_header(file):
    if not validate_file(file):
        raise ValueError(f"Invalid file (missing or empty): {file}")

    try:
        result = subprocess.run(['save2gdf', '-JSON', file], stdout=subprocess.PIPE, stderr=subprocess.PIPE);
        
        if result.returncode == 0:
            try:
                raw_output = result.stdout.decode(errors='ignore').replace('\n','').replace('\r','').replace('\t','').replace('inf','0')
                out = json.loads(raw_output)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON header from {file}. Error: {e}")
        else:
            err_msg = result.stderr.decode(errors='ignore').strip()
            raise ValueError(f"save2gdf failed on {file}. Return code: {result.returncode}. Msg: {err_msg}");
            
    except FileNotFoundError:
        raise FileNotFoundError('The required command `save2gdf` seems not appearing in your environment variables.');
    return out;

def get_start(file):
    header = read_header(file)
    if header is None or 'StartOfRecording' not in header:
        raise ValueError(f"Header missing 'StartOfRecording' field in file: {file}")
    str_start = header['StartOfRecording']
    out = parse_timestamp(str_start)
    if out is None:
        raise ValueError(f"Could not parse StartOfRecording timestamp: '{str_start}' in file {file}")
    return out

def time_delta(label_timestamp, start):
    if not isinstance(label_timestamp, datetime) or not isinstance(start, datetime):
         return 0
    out = (label_timestamp - start).total_seconds()
    return int(out);

def get_events(file):
    out = [];
    header = read_header(file)
    if 'EVENT' not in header:
         print(f"Warning: No EVENT data found in {file}")
         return []

    events = header['EVENT'];
    for event in events:
        time_obj = parse_timestamp(event['TimeStamp'])
        if time_obj is None:
            continue

        stage = None;
        description = event.get('Description', '')
        if re.search('1$', description) is not None: stage = '1';
        elif re.search('2$', description) is not None: stage = '2';
        elif re.search('3$', description) is not None: stage = '3';
        elif re.search('4$', description) is not None: stage = '4';
        elif re.search('W$', description) is not None: stage = 'W';
        elif re.search('R$', description) is not None: stage = 'R';
        elif re.search('M$', description) is not None: stage = 'M';
        else: stage = 'L';
        out.append((time_obj, stage));
    return out;

def relative(start, events):
    out = [];
    for event in events:
        if event[0] is None: continue
        delta = time_delta(event[0], start)
        out.append([delta, event[0], event[1]]);
    out = numpy.array(out);
    return out;

def save(path, ori_fname, array):
    if array.size == 0:
        print(f"Warning: Result array for {ori_fname} is empty. Skipping save.")
        return
    fname = os.path.join(path, os.path.splitext(ori_fname)[0] + '.npy');
    numpy.save(fname, array);
    print(f"Successfully processed: {ori_fname}")

def main():
    usage = 'usage: %prog [options] SOURCE DEST';
    parser = OptionParser(usage = usage);
    (options, args) = parser.parse_args();
    
    if len(args) != 2:
        parser.print_usage(file = sys.stderr);
        sys.stderr.write("\tUse '-h' for help\n\n");
        sys.exit(1);
    else:
        source, dest = args;
        
        # --- 路径检查与自动创建 ---
        if not os.path.isdir(source):
            raise ValueError(f'Source directory "{source}" does not exist.');
        
        if not os.path.exists(dest):
            print(f"Target directory '{dest}' does not exist. Creating it now...")
            os.makedirs(dest, exist_ok=True)
        elif not os.path.isdir(dest):
            raise ValueError(f'Target path "{dest}" exists but is not a directory.');
        # -----------------------

        try:
            data_entries = get_list(recognize_edf_or_edfx(source), source);
        except Exception as e:
            print(f"Error identifying dataset structure: {e}")
            sys.exit(1)

        print(f"Found {len(data_entries)} pairs of files to process.")
        damaged_files = []

        for entry in data_entries:
            rec, hyp = entry;
            rec_path_full = os.path.join(source, rec)
            hyp_path_full = os.path.join(source, hyp)
            
            try:
                print(f"Processing pair: {rec} | {hyp}")

                if not validate_file(rec_path_full):
                    damaged_files.append((rec, "Recording file missing or empty"))
                    continue
                if not validate_file(hyp_path_full):
                    damaged_files.append((hyp, "Hypnogram file missing or empty"))
                    continue
                
                start = get_start(rec_path_full);
                events = get_events(hyp_path_full);
                
                if len(events) == 0:
                    damaged_files.append((hyp, "No valid events found (Empty list)"))
                    continue
                    
                array = relative(start, events);
                save(dest, rec, array);
                
            except ValueError as ve:
                print(f"Error [Invalid Data] processing {rec}: {ve}")
                damaged_files.append((f"{rec} / {hyp}", str(ve)))
                continue 
            except Exception as e:
                print(f"Error [Unexpected] processing {rec}: {e}")
                damaged_files.append((f"{rec} / {hyp}", str(e)))
                continue 

        # 打印报告
        print("\n" + "="*60)
        print(f"{'DAMAGED / SKIPPED FILES REPORT':^60}")
        print("="*60)
        
        if len(damaged_files) == 0:
            print("Congratulations! No damaged files detected.")
        else:
            print(f"Total damaged/skipped entries: {len(damaged_files)}\n")
            print(f"{'File Name':<35} | {'Reason'}")
            print("-" * 60)
            for file_name, reason in damaged_files:
                clean_reason = reason.replace('\n', ' ').strip()
                if len(clean_reason) > 50: clean_reason = clean_reason[:47] + "..."
                print(f"{file_name:<35} | {clean_reason}")
        
        print("="*60 + "\n")

if __name__ == '__main__':
    try:
        main();
    except KeyboardInterrupt:
        sys.exit(1);