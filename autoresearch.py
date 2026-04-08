#!/usr/bin/env python3
"""
AutoResearch Loop - Karpathy-style autonomous experimentation.
Runs continuously, trying different strategies, logging results,
and adapting based on feedback.

This script is the DRIVER. Claude Code runs this and it orchestrates
the experimentation cycle.
"""

import os
import sys
import json
import time
import subprocess
import datetime

PROJECT_DIR = "/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge"
LOG_FILE = os.path.join(PROJECT_DIR, "experiment_log.md")
STRATEGY_FILE = os.path.join(PROJECT_DIR, "strategy.md")
BEST_FILE = os.path.join(PROJECT_DIR, "best_result.json")

def log_experiment(exp_id, description, accuracy, details, duration_sec):
    """Append experiment result to log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Experiment {exp_id} [{timestamp}]\n")
        f.write(f"- **Description**: {description}\n")
        f.write(f"- **Accuracy**: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"- **Duration**: {duration_sec:.0f}s\n")
        f.write(f"- **Details**: {details}\n")
        f.write(f"---\n")

def load_best():
    """Load best result so far."""
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE) as f:
            return json.load(f)
    return {"accuracy": 0.0, "experiment_id": 0, "description": "none"}

def save_best(exp_id, accuracy, description):
    """Save new best result."""
    with open(BEST_FILE, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "experiment_id": exp_id,
            "description": description,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=2)

def git_push(message):
    """Push results to GitHub."""
    try:
        os.chdir(PROJECT_DIR)
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
        subprocess.run(["git", "push"], check=True, capture_output=True)
        print(f"[GIT] Pushed: {message}")
    except subprocess.CalledProcessError:
        print("[GIT] Nothing to push or push failed")

def init_log():
    """Initialize experiment log if not exists."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("# Experiment Log\n\n")
            f.write("Automated experiment tracking for COMP3314 Image Classification.\n\n")
            f.write("| Exp | Accuracy | Description | Time |\n")
            f.write("|-----|----------|-------------|------|\n")

if __name__ == "__main__":
    print("=" * 60)
    print("AutoResearch Loop - COMP3314 Image Classification")
    print("=" * 60)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Log file: {LOG_FILE}")
    print()
    
    init_log()
    best = load_best()
    print(f"Current best accuracy: {best['accuracy']:.4f}")
    print()
    print("This script provides the framework.")
    print("Claude Code should import and use these functions")
    print("to log experiments and track progress.")
    print()
    print("Functions available:")
    print("  log_experiment(id, desc, acc, details, duration)")
    print("  load_best() -> dict")
    print("  save_best(id, acc, desc)")
    print("  git_push(message)")
