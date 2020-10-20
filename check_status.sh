#!/bin/bash

USERNAME=${1:-"gashwin1"}
OUTPUT_FOLDER=${2:"output/"}
STATUS="status/"

# Check all jobs.
echo "Number of Running jobs: `squeue | grep $USERNAME | wc -l`"
squeue | grep $USERNAME > $STATUS/current.jobs
echo " " >> $STATUS/current.jobs

# Maintain a list of jobs.
sacct --format="JobID,JobName%40" >> $STATUS/current.jobs

# Check for errors.
echo "Number of Job Errors: `grep -sir "error" output/ | wc -l`"
grep -sir "error" output/ > $STATUS/slurm.error

# Check for errors.
echo "Number of Result BDMA (Evaluation) Errors: `grep -sir "error" data/results/*.results | wc -l`"
grep -sir "error" data/results/*.results > $STATUS/bdma.jobs.error

# Check for MUSE baseline errors.
echo "Number of MUSE Baseline Result (Evaluation) Errors: `grep -sir "error" data/muse_baseline_results/*.results | wc -l`"
grep -sir "error" data/muse_baseline_results/*.results > $STATUS/baseline.jobs.error

# Check disk usage.
echo "Total disk space left..."
df -h | grep ferraro
