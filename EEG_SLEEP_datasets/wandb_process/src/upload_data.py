import json
import wandb

# 1. 初始化你的 wandb 实验
wandb.init(project="EEG-VLM_detail", name="run_eeg_vlm_G2_320e4a1b32_1")

# 2. 指定你的 trainer_state.json 文件路径
# 请将这里的路径替换为你电脑上该文件的实际路径
json_path = "/liuran/liuran/EEG/EEG-VLM-main/checkpoints/run_eeg_vlm_G2_320e4a1b32_1/checkpoint-320/trainer_state.json"

# 3. 读取 json 文件
with open(json_path, "r", encoding="utf-8") as f:
    state_data = json.load(f)

# 4. 获取日志历史记录
log_history = state_data.get("log_history", [])

# 5. 遍历并上传数据画曲线
for log in log_history:
    # 只要这行 log 里有 step，我们就把它当做 x 轴参考并记录所有指标
    if "step" in log:
        wandb.log(log)

# 6. 结束运行
wandb.finish()