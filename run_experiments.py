import subprocess
import re

experiments = [
    {
        "config": "early_catfuse.yaml",
        "fusion_type": "early",
    },
    {
        "config": "late_catfuse.yaml",
        "fusion_type": "late",
    },
    {
        "config": "early_cbamc.yaml",
        "fusion_type": "early",
    },
    {
        "config": "late_cbamc.yaml",
        "fusion_type": "late",
    },
    {
        "config": "early_transenc.yaml",
        "fusion_type": "early",
    }
]

csv_header = "Config,Fusion Type,Total Epochs,Best Epoch,Time Taken (hours)\n"

for experiment, idx in enumerate(experiments):
    config = experiment["config"]
    fusion_type = experiment["fusion_type"]
    batch_size = 4 if fusion_type == "late" else 16    # Adjust batch size as needed

    print(f"Running experiment with config: {config}, fusion type: {fusion_type}")
    
    res = subprocess.run(["python3", "train.py", "--weights", "''", "--cfg", config, "--data", "DeepSDO.yaml", "--epochs", "1000", "--imgsz", "512", "--fusion", "--fusion-type", fusion_type, "--tl-fusion", "--batch-size", batch_size, "--save-period", "100"], stdout=subprocess.PIPE).stdout.decode('utf-8')
    
    total_epochs = None
    best_epoch = None
    time_taken = None

    for line in res.split('\n'):
        if line.startswith("Stopping training early as no improvement observed in last"):
            match = re.search(r'Best results observed at epoch (\d+)', line)
            if match:
                best_epoch = int(match.group(1))
        elif "epochs completed in" in line:
            match = re.search(r'(\d+) epochs completed in ([\d\.]+)', line)
            if match:
                total_epochs = int(match.group(1))
                time_taken = float(match.group(2))

    # write results to a file
    with open("experiment_results.csv", "a") as f:
        if idx == 0:
            f.write(csv_header)
        if total_epochs is not None and best_epoch is not None and time_taken is not None:
            f.write(f"{config},{fusion_type},{total_epochs},{best_epoch},{time_taken}\n")
        else:
            print(f"Failed to parse results for config: {config}, fusion type: {fusion_type}")