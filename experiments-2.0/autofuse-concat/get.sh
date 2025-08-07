#!/bin/bash

# === Path to the sweep_times CSV ===
sweep_csv="sweep_times.csv"

# === Print Markdown Header ===
echo "| Dataset      | Train Time (s) | Inference Time (s) | XGB Total (s) | Sweep Runtime (s) | Total Time (s) |"
echo "|--------------|----------------|---------------------|---------------|-------------------|----------------|"

# === Loop over each result CSV ===
for file in results_*_224.csv; do
  dataset=$(basename "$file" | sed 's/results_\(.*\)_224\.csv/\1/')

  # Get XGB times
  train_sum=$(awk -F',' 'NR>1 {sum+=$11} END {printf "%.2f", sum}' "$file")
  infer_sum=$(awk -F',' 'NR>1 {sum+=$12} END {printf "%.2f", sum}' "$file")
  xgb_total=$(awk -F',' 'NR>1 {sum+=$11+$12} END {printf "%.2f", sum}' "$file")

  # Get sweep time from CSV
  sweep_time=$(awk -F',' -v d="$dataset" 'tolower($1)==tolower(d) {printf "%.2f", $2}' "$sweep_csv")

  # Compute grand total
  total=$(awk -v x="$xgb_total" -v s="$sweep_time" 'BEGIN {printf "%.2f", x + s}')

  # Print row
  printf "| %-12s | %14s | %19s | %13s | %17s | %14s |\n" "$dataset" "$train_sum" "$infer_sum" "$xgb_total" "$sweep_time" "$total"
done
