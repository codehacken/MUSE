#!/bin/bash

USERNAME=${1:-"gashwin1"}
OUTPUT_FOLDER=${2:"output/"}

# Check all jobs.
echo "List of running jobs..."
squeue | grep $USERNAME
echo "Number of Running jobs: `squeue | grep $USERNAME | wc -l`"
echo " "

# Check for errors.
echo "Number of Errors: `grep -sir "error" output/ | wc -l`"
echo "List of errors found..."
grep -sir "error" output/

# Check disk usage.
echo "Total disk space left..."
df -h | grep ferraro
