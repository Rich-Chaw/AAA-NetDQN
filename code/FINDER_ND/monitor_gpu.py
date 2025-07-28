#!/usr/bin/env python3
"""
GPU Monitoring Script for GraphDQN Training
"""
import time
import subprocess
import threading
import os

def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    util, mem_used, mem_total = line.split(', ')
                    return int(util), int(mem_used), int(mem_total)
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return 0, 0, 0

def monitor_gpu():
    """Monitor GPU usage continuously"""
    print("Starting GPU monitoring...")
    print("Time\t\tGPU Util%\tMemory Used\tMemory Total\tMemory%")
    print("-" * 70)
    
    start_time = time.time()
    while True:
        try:
            util, mem_used, mem_total = get_gpu_utilization()
            elapsed = time.time() - start_time
            mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
            
            print(f"{elapsed:8.1f}s\t{util:8d}%\t{mem_used:10d}MB\t{mem_total:11d}MB\t{mem_percent:6.1f}%")
            
            time.sleep(2)  # Update every 2 seconds
        except KeyboardInterrupt:
            print("\nGPU monitoring stopped.")
            break
        except Exception as e:
            print(f"Error in monitoring: {e}")
            break

if __name__ == "__main__":
    monitor_gpu() 