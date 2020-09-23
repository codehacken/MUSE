#!/bin/bash

USERNAME=${1:-"gashwin1"}
OUTPUT_FOLDER=${2:"output/"}

# Check all jobs.
echo "List of running jobs..."
squeue | grep $USERNAME
echo "Number of Running jobs: `squeue | grep $USERNAME | wc -l`"
echo " "

# Check for errors.
echo "Number of Job Errors: `grep -sir "error" output/ | wc -l`"
echo "List of errors found..."
grep -sir "error" output/

# Check for errors.
echo "Number of Result (Evaluation) Errors: `grep -sir "error" data/results/*.results/ | wc -l`"
echo "List of errors found..."
grep -sir "error" data/results/*.results

# Check for MUSE baseline errors.
echo "Number of MUSE Baseline Result (Evaluation) Errors: `grep -sir "error" data/muse_baseline_results/*.results/ | wc -l`"
echo "List of errors found..."
grep -sir "error" data/muse_baseline_results/*.results

# Check disk usage.
echo "Total disk space left..."
df -h | grep ferraro
