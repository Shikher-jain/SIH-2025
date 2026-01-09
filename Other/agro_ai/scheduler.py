# scheduler.py
"""
Run this script to start an in-process scheduler that triggers weekly training.
Recommended: run as a system service or put a cron job to run this script at startup.
"""
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def weekly_train():
    print("Weekly training started...")
    # call main.py which runs dataset build + train
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "main.py")], check=False)
    print("Weekly training finished.")

if __name__ == "__main__":
    sched = BlockingScheduler()
    # schedule every 7 days at 02:00 AM server time:
    sched.add_job(weekly_train, 'interval', days=7, next_run_time=None)
    print("Scheduler started — weekly training job added (every 7 days).")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
