#!/bin/bash

sweep_csv="sweep_times_self_attention.csv"

echo "| Dataset       | Train Time (s) | Inference Time (s) | XGB Total (s) | Sweep Runtime (s) | Total Time (s) |"
echo "|---------------|----------------|---------------------|----------------|-------------------|----------------|"

for dataset in bloodmnist breastmnist chestmnist dermamnist octmnist organamnist organcmnist organsmnist pathmnist pneumoniamnist retinamnist tissuemnist; do
  total_train=0
  total_infer=0
  total_xgb=0

  for proj in 256 512 768; do
    file="autofuse-self-attention-$proj/results_${dataset}_224.csv"
    [ ! -f "$file" ] && continue

    train=$(awk -F',' 'NR>1 {sum+=$11} END {print sum}' "$file")
    infer=$(awk -F',' 'NR>1 {sum+=$12} END {print sum}' "$file")
    xgb=$(awk -F',' 'NR>1 {sum+=$11+$12} END {print sum}' "$file")

    total_train=$(awk -v a="$total_train" -v b="$train" 'BEGIN {print a + b}')
    total_infer=$(awk -v a="$total_infer" -v b="$infer" 'BEGIN {print a + b}')
    total_xgb=$(awk -v a="$total_xgb" -v b="$xgb" 'BEGIN {print a + b}')
  done

  # 🚑 Strip carriage returns, whitespace, and make sure it's float-safe
  sweep_time=$(awk -F',' -v d="$dataset" 'tolower($1)==tolower(d) {print $2}' "$sweep_csv" | tr -d '\r' | xargs)
  [ -z "$sweep_time" ] && sweep_time=0

  # 🧠 Ensure all vars are clean floats
  grand_total=$(awk -v x="$total_xgb" -v s="$sweep_time" 'BEGIN {printf "%.2f", x + s}')
  total_train=$(awk -v x="$total_train" 'BEGIN {printf "%.2f", x}')
  total_infer=$(awk -v x="$total_infer" 'BEGIN {printf "%.2f", x}')
  total_xgb=$(awk -v x="$total_xgb" 'BEGIN {printf "%.2f", x}')
  sweep_time=$(awk -v x="$sweep_time" 'BEGIN {printf "%.2f", x}')

  printf "| %-13s | %14s | %19s | %14s | %17s | %14s |\n" \
    "$dataset" "$total_train" "$total_infer" "$total_xgb" "$sweep_time" "$grand_total"
done