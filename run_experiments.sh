#!/bin/bash

experiments=(
  "early_catfuse.yaml early"
  "late_catfuse.yaml late"
  "early_cbamc.yaml early"
  "late_cbamc.yaml late"
  "early_transenc.yaml early"
)

csv_header="Config,Fusion Type,Total Epochs,Best Epoch,Time Taken (hours),Experiment Path"
results_file="experiment_results.csv"

# Write header if file doesn't exist
if [ ! -f "$results_file" ]; then
  echo "$csv_header" > "$results_file"
fi

for exp in "${experiments[@]}"; do
  config=$(echo $exp | awk '{print $1}')
  fusion_type=$(echo $exp | awk '{print $2}')
  batch_size=4

  echo "Running experiment with config: $config, fusion type: $fusion_type"

  tmp_output="output_${config}_${fusion_type}.log"
  python3 ./train.py --weights '' --cfg "$config" --data DeepSDO.yaml \
    --epochs 1000 --imgsz 512 --fusion --fusion-type "$fusion_type" \
    --tl-fusion --batch-size "$batch_size" --save-period 100 --cache ram 2>&1 | tee "$tmp_output"

  total_epochs=$(echo "$output" | grep "epochs completed in" | sed -n 's/^\([0-9]\+\) epochs completed in \([0-9.]\+\).*/\1/p' | tail -1)
  time_taken=$(echo "$output" | grep "epochs completed in" | sed -n 's/^[0-9]\+ epochs completed in \([0-9.]\+\).*/\1/p' | tail -1)
  best_epoch=$(echo "$output" | grep "Stopping training early as no improvement observed in last" | sed -n 's/.*Best results observed at epoch \([0-9]\+\).*/\1/p' | tail -1)
  exp_path=$(echo "$output" | grep "Results saved to" | sed -n 's/Results saved to \(runs\/train\/exp[0-9]\+\).*/\1/p' | tail -1)

  if [[ -n "$total_epochs" && -n "$best_epoch" && -n "$time_taken" ]]; then
    echo "$config,$fusion_type,$total_epochs,$best_epoch,$time_taken,$exp_path" >> "$results_file"
  else
    echo "Failed to parse results for config: $config, fusion type: $fusion_type"
  fi

  rm -f "$tmp_output"
done