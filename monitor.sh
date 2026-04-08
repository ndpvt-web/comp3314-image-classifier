#!/bin/bash
# Monitor and auto-push script for COMP3314 project
# Runs every 5 minutes via cron, pushes any new changes to GitHub

PROJECT_DIR="/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge"
MONITOR_LOG="$PROJECT_DIR/monitor.log"

cd "$PROJECT_DIR" || exit 1

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Check for changes
CHANGES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

if [ "$CHANGES" -gt "0" ]; then
    echo "[$TIMESTAMP] Found $CHANGES changed files, pushing..." >> "$MONITOR_LOG"
    
    git add -A
    git commit -m "Auto-update: experiment results and progress ($TIMESTAMP)" 2>/dev/null
    git push 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "[$TIMESTAMP] Push successful" >> "$MONITOR_LOG"
    else
        echo "[$TIMESTAMP] Push failed" >> "$MONITOR_LOG"
    fi
else
    echo "[$TIMESTAMP] No changes to push" >> "$MONITOR_LOG"
fi

# Log current best result if exists
if [ -f "$PROJECT_DIR/best_result.json" ]; then
    BEST=$(python3 -c "import json; d=json.load(open('$PROJECT_DIR/best_result.json')); print(f'Best: {d[\"accuracy\"]:.4f} - {d[\"description\"]}')" 2>/dev/null)
    echo "[$TIMESTAMP] $BEST" >> "$MONITOR_LOG"
fi

# Log experiment count
if [ -f "$PROJECT_DIR/experiment_log.md" ]; then
    EXP_COUNT=$(grep -c "^## Experiment" "$PROJECT_DIR/experiment_log.md" 2>/dev/null)
    echo "[$TIMESTAMP] Total experiments: $EXP_COUNT" >> "$MONITOR_LOG"
fi
