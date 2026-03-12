# -*- coding: utf-8 -*-
"""
Sleep-EDF Advanced Processor (Final Strict Version)
Features:
1. Filter: Fpz-Cz, 0.5-35Hz, IIR, Butterworth Order 1.
2. Outputs: NPY, Raw Plot, Filtered Plot, 30s Epoch Plots.
3. Validation: Verify ALL outputs immediately. Delete corrupt files.
4. Logging: Write errors to 'error_log.txt'.
5. Smart Skip: Resume processing if valid base files exist.
"""
import os
import re
import sys
import util
import json
import numpy as np
import subprocess
import matplotlib
# Force Agg backend to prevent memory leaks or no-GUI errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from optparse import OptionParser


"""
    运行：
    python3 annotation_convertor_filter_split30s_img.py /liuran/liuran/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette  /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/npy  /liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_img
"""

try:
    import mne
except ImportError:
    print("Error: Missing mne library. Please run `pip install mne`.")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Missing Pillow library. Please run `pip install pillow`.")
    sys.exit(1)

# --- Configuration Paths ---
FILTER_IMG_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_filter_img"
SPLIT_IMG_DIR = "/liuran/liuran/EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event/fpz-cz_split30s_img"
ERROR_LOG_FILE = "error_log.txt"

# --- Helper Functions ---

def log_error(filename, message):
    """Log errors to file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] File: {filename} | Error: {message}\n"
    print(f"  [Log Error] {message}")
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_msg)
    except:
        pass

def validate_file(filepath):
    """Check input file existence"""
    if not os.path.exists(filepath): return False, f"Not found: {filepath}"
    if os.path.getsize(filepath) == 0: return False, "Empty file"
    return True, "Valid"

def verify_image_integrity(image_path):
    """Strictly verify image file using PIL"""
    if not os.path.exists(image_path): return False, "Missing"
    try:
        with Image.open(image_path) as img:
            img.verify() # Attempts to read file header/structure
        return True, "Valid"
    except Exception as e:
        return False, f"Corrupt: {e}"

def verify_npy_integrity(npy_path):
    """Strictly verify NPY file using numpy load"""
    if not os.path.exists(npy_path): return False, "Missing"
    try:
        # allow_pickle=True is required for object arrays (mixed types)
        data = np.load(npy_path, allow_pickle=True)
        if data.size == 0: return False, "Empty array"
        return True, "Valid"
    except Exception as e:
        return False, f"Corrupt: {e}"

def check_skip_condition(filename_base, dest_npy, dest_img):
    """Check if base files (NPY + Raw Image) exist and are valid"""
    target_npy = os.path.join(dest_npy, filename_base + '.npy')
    target_img = os.path.join(dest_img, filename_base + '.png')

    valid_npy, msg_npy = verify_npy_integrity(target_npy)
    if not valid_npy: return False, f"NPY invalid: {msg_npy}"

    valid_img, msg_img = verify_image_integrity(target_img)
    if not valid_img: return False, f"Raw Image invalid: {msg_img}"

    return True, "Valid base files exist"

# --- Time & Data Parsing ---

def parse_timestamp(time_str):
    dt_out = None; add_delta = timedelta(0)
    if ' 24:' in time_str: time_str = time_str.replace(' 24:', ' 00:'); add_delta += timedelta(days=1)
    if re.search(r':60(?=\.|$)', time_str): time_str = re.sub(r':60(?=\.|$)', ':00', time_str, count=1); add_delta += timedelta(minutes=1)
    time_formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%b-%d %H:%M:%S', '%Y-%b-%d %H:%M:%S.%f']
    for fmt in time_formats:
        try: dt_out = datetime.strptime(time_str, fmt); break
        except ValueError: continue
    if dt_out is not None: dt_out += add_delta
    return dt_out

def recognize_edf_or_edfx(path):
    if len(util.search_file(path, '.edf')[1]) != 0: return False
    elif len(util.search_file(path, '.rec')[1]) != 0: return True
    else: raise ValueError('Not Sleep-EDF directory.')

def get_list(is_sleep_edf, path):
    if is_sleep_edf:
        rec = util.search_file(path, '0.rec$')[1]; rec.sort()
        hyp = util.search_file(path, '0.hyp$')[1]; hyp.sort()
    else:
        rec = util.search_file(path, '-PSG.edf$')[1]; rec.sort()
        hyp = util.search_file(path, '-Hypnogram.edf$')[1]; hyp.sort()
    return list(zip(rec[:min(len(rec), len(hyp))], hyp[:min(len(rec), len(hyp))]))

def read_header(file):
    try:
        result = subprocess.run(['save2gdf', '-JSON', file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return json.loads(result.stdout.decode(errors='ignore').replace('\n','').replace('\r','').replace('\t','').replace('inf','0'))
    except: pass
    return None

def get_start(file):
    header = read_header(file)
    if not header: raise ValueError(f"Header read failed: {file}")
    return parse_timestamp(header['StartOfRecording'])

def time_delta(label_timestamp, start):
    return int((label_timestamp - start).total_seconds())

def get_events(file):
    out = []; header = read_header(file)
    if not header or 'EVENT' not in header: return []
    for event in header['EVENT']:
        t = parse_timestamp(event['TimeStamp'])
        if t is None: continue
        desc = event.get('Description', '')
        if re.search('1$', desc): stage='1'
        elif re.search('2$', desc): stage='2'
        elif re.search('3$', desc): stage='3'
        elif re.search('4$', desc): stage='4'
        elif re.search('W$', desc): stage='W'
        elif re.search('R$', desc): stage='R'
        elif re.search('M$', desc): stage='M'
        else: stage='L'
        out.append((t, stage))
    return out

def relative(start, events):
    out = []
    for event in events: out.append([time_delta(event[0], start), event[0], event[1]])
    return np.array(out)

# --- Core Processing Functions ---

def save_npy(path, ori_fname, array):
    """Save NPY and IMMEDIATELY verify it"""
    fname = os.path.join(path, os.path.splitext(ori_fname)[0] + '.npy')
    try:
        np.save(fname, array)
        # [STRICT VALIDATION]
        is_valid, msg = verify_npy_integrity(fname)
        if not is_valid:
            if os.path.exists(fname): os.remove(fname)
            log_error(fname, f"NPY Validation Failed: {msg}")
            raise ValueError(f"Generated NPY corrupt: {msg}")
        print(f"Saved Valid NPY: {fname}")
    except Exception as e:
        if os.path.exists(fname): os.remove(fname)
        log_error(ori_fname, f"Save NPY Error: {e}")
        raise e

def plot_and_save_image(rec_path, npy_array, output_img_dir, filename):
    """Raw Plot + Verify"""
    try:
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose='ERROR')
        target_ch = next((ch for ch in raw.ch_names if "Fpz-Cz" in ch), None)
        if not target_ch: return
        data, times = raw.copy().pick([target_ch])[:]
        data = data[0] * 1e6
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(times, data, color='k', linewidth=0.5); ax1.set_title(f"{target_ch} - {filename}")
        
        stage_map = {'W': 0, 'R': -1, '1': -2, '2': -3, '3': -4, '4': -4, 'M': 1, 'L': 1}
        hyp_y = [stage_map.get(s, 1) for s in npy_array[:, 2]]
        onset = npy_array[:, 0].astype(float)
        ax2.step(onset, hyp_y, where='post', color='navy')
        ax2.set_yticks([1, 0, -1, -2, -3, -4])
        ax2.set_xlim(0, times[-1])
        
        out = os.path.join(output_img_dir, os.path.splitext(filename)[0] + '.png')
        plt.tight_layout(); plt.savefig(out, dpi=100); plt.close(fig)
        
        # [STRICT VALIDATION]
        is_valid, msg = verify_image_integrity(out)
        if not is_valid:
            if os.path.exists(out): os.remove(out)
            log_error(out, f"Raw Image Validation Failed: {msg}")
            raise ValueError(f"Raw Image Corrupt: {msg}")
        print(f"Saved Valid Raw Image: {out}")
    except Exception as e:
        log_error(filename, f"Raw plot failed: {e}")
        raise e

def save_filtered_full_night_image(rec_path, npy_array, filename):
    """Filtered Plot (Correct Filter Logic) + Verify"""
    try:
        if not os.path.exists(FILTER_IMG_DIR): os.makedirs(FILTER_IMG_DIR)
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose='ERROR')
        target_ch = next((ch for ch in raw.ch_names if "Fpz-Cz" in ch), None)
        if target_ch:
            # --- [TRANSPLANTED FILTER LOGIC] ---
            # 0.5-35Hz, IIR, Butterworth Order 1
            print(f"  Filtering Full-Night {filename} (0.5-35Hz)...")
            raw.filter(l_freq=0.5, h_freq=35.0, picks=[target_ch], method='iir', 
                       iir_params={'order': 1, 'ftype': 'butter'}, verbose=False)
            # -----------------------------------
            
            data, times = raw[target_ch]; data = data[0] * 1e6
            if np.isnan(data).any(): 
                log_error(filename, "Filtered data contains NaN"); return

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(times, data, color='k', linewidth=0.5); ax1.set_title(f"Filtered {target_ch}")
            
            stage_map = {'W': 0, 'R': -1, '1': -2, '2': -3, '3': -4, '4': -4, 'M': 1, 'L': 1}
            hyp_y = [stage_map.get(s, 1) for s in npy_array[:, 2]]
            onset = npy_array[:, 0].astype(float)
            ax2.step(onset, hyp_y, where='post', color='navy')
            ax2.set_yticks([1, 0, -1, -2, -3, -4])
            ax2.set_xlim(0, times[-1])
            
            out = os.path.join(FILTER_IMG_DIR, os.path.splitext(filename)[0] + '_filtered.png')
            plt.tight_layout(); plt.savefig(out, dpi=100); plt.close(fig)
            
            # [STRICT VALIDATION]
            is_valid, msg = verify_image_integrity(out)
            if not is_valid:
                if os.path.exists(out): os.remove(out)
                log_error(out, f"Filtered Image Validation Failed: {msg}")
            else:
                print(f"  Saved Valid Filtered Full-Night: {out}")
    except Exception as e:
        log_error(filename, f"Filtered plot crash: {e}")

def save_filtered_30s_epochs(rec_path, npy_array, filename):
    """Filtered 30s Split (Correct Filter Logic) + Verify Each"""
    try:
        if not os.path.exists(SPLIT_IMG_DIR): os.makedirs(SPLIT_IMG_DIR)
        
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose='ERROR')
        target_ch = next((ch for ch in raw.ch_names if "Fpz-Cz" in ch), None)
        if not target_ch: return

        print(f"  Generating 30s epochs for {filename} ...")
        
        # --- [TRANSPLANTED FILTER LOGIC] ---
        raw.filter(l_freq=0.5, h_freq=35.0, picks=[target_ch], method='iir', 
                   iir_params={'order': 1, 'ftype': 'butter'}, verbose=False)
        # -----------------------------------
        
        signal_data = raw[target_ch][0][0] * 1e6 
        sfreq = raw.info['sfreq']

        if np.isnan(signal_data).any() or np.isinf(signal_data).any():
            log_error(filename, "Signal contains NaN/Inf. Skipping splits.")
            return

        event_onsets = npy_array[:, 0].astype(int)
        event_stages = npy_array[:, 2]
        epoch_sec = 30
        samples_per_epoch = int(epoch_sec * sfreq)
        total_epochs = len(signal_data) // samples_per_epoch
        base_name = os.path.splitext(filename)[0]

        corrupt_count = 0

        for i in range(total_epochs):
            start_sec = i * epoch_sec
            start_idx = int(start_sec * sfreq)
            end_idx = int((start_sec + epoch_sec) * sfreq)
            
            epoch_data = signal_data[start_idx:end_idx]
            if len(epoch_data) < samples_per_epoch: continue
            if np.all(epoch_data == 0): continue

            epoch_time = np.linspace(0, epoch_sec, len(epoch_data))
            
            valid_indices = np.where(event_onsets <= start_sec)[0]
            current_stage = event_stages[valid_indices[-1]] if len(valid_indices) > 0 else "?"
            
            fig = plt.figure(figsize=(10, 4))
            plt.plot(epoch_time, epoch_data, color='k', linewidth=0.8)
            plt.title(f"{base_name} | Ep:{i+1} | {current_stage}")
            plt.grid(True, alpha=0.3); plt.xlim(0, epoch_sec)
            plt.ylabel("uV"); plt.xlabel("sec")
            
            save_name = f"{base_name}_epoch{i+1}_stage{current_stage}.png"
            save_path = os.path.join(SPLIT_IMG_DIR, save_name)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close(fig)

            # [STRICT VALIDATION]
            is_valid, msg = verify_image_integrity(save_path)
            if not is_valid:
                if os.path.exists(save_path): os.remove(save_path) # Delete bad file
                log_error(save_name, f"Corrupt 30s Image: {msg}") # Log error
                corrupt_count += 1

            if (i+1) % 200 == 0: 
                print(f"    Processed {i+1}/{total_epochs} epochs...")

        if corrupt_count > 0:
            print(f"  [Warning] {corrupt_count} images were corrupt and deleted for {filename}")
        else:
            print(f"  Finished {filename}: All {total_epochs} epochs valid.")

    except Exception as e: 
        log_error(filename, f"Split 30s Epochs Crash: {e}")
        print(f"  Split Error: {e}")

def main():
    usage = 'usage: %prog [options] SOURCE NPY_DEST IMG_DEST'
    parser = OptionParser(usage = usage)
    (options, args) = parser.parse_args()
    
    if len(args) != 3: print("Need 3 args"); sys.exit(1)
    
    source, dest_npy, dest_img = args
    if not os.path.exists(dest_npy): os.makedirs(dest_npy)
    if not os.path.exists(dest_img): os.makedirs(dest_img)

    data_entries = get_list(recognize_edf_or_edfx(source), source)
    print(f"Found {len(data_entries)} pairs.")

    with open(ERROR_LOG_FILE, "a") as f:
        f.write(f"\n--- New Run Started: {datetime.now()} ---\n")

    for entry in data_entries:
        rec, hyp = entry
        full_rec = os.path.join(source, rec)
        full_hyp = os.path.join(source, hyp)
        fname_base = os.path.splitext(rec)[0]

        # 1. Smart Skip Check (Validates existing base files)
        skip_basic, skip_reason = check_skip_condition(fname_base, dest_npy, dest_img)

        if skip_basic:
            print(f"Basic files valid for {rec}. Checking secondary outputs...")
            try:
                # Load Valid NPY
                target_npy = os.path.join(dest_npy, fname_base + '.npy')
                array = np.load(target_npy, allow_pickle=True)
                
                # Check/Generate secondary outputs
                # (Functions handle file creation and validation internally)
                save_filtered_30s_epochs(full_rec, array, rec)
                save_filtered_full_night_image(full_rec, array, rec)
                continue
            except Exception as e:
                log_error(rec, f"Supplemental generation failed: {e}")
                print(f"Error loading existing data: {e}")

        # Full Pipeline (If base files missing or invalid)
        print(f"Full processing: {rec} (Reason: {skip_reason}) ...")
        try:
            start = get_start(full_rec)
            events = get_events(full_hyp)
            if not events: 
                log_error(rec, "No events in hypnogram")
                continue
            array = relative(start, events)
            
            # Save & Verify ALL
            save_npy(dest_npy, rec, array)
            plot_and_save_image(full_rec, array, dest_img, rec)
            save_filtered_full_night_image(full_rec, array, rec)
            save_filtered_30s_epochs(full_rec, array, rec)
            print("[Done]")
        except Exception as e:
            log_error(rec, f"Fatal Pipeline Error: {e}")
            print(f"[Failed] {rec}: {e}")

if __name__ == '__main__':
    main()