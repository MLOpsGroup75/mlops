#!/usr/bin/env python3
"""
Simple script to fix basic linting issues automatically
"""
import subprocess
import os

def run_command(cmd, check=True):
    """Run a command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error running: {cmd}")
            print(f"Error: {result.stderr}")
        return result
    except Exception as e:
        print(f"Exception running {cmd}: {e}")
        return None

def fix_basic_issues():
    """Fix basic formatting issues"""
    print("🔧 Auto-fixing basic linting issues...")
    
    # Auto-format with black (this will fix most formatting issues)
    print("1️⃣ Running black auto-formatter...")
    run_command("black services/api/ services/predict/", check=False)
    
    # Auto-fix import sorting with isort  
    print("2️⃣ Fixing import sorting...")
    run_command("isort services/api/ services/predict/", check=False)
    
    print("✅ Basic fixes completed!")

if __name__ == "__main__":
    fix_basic_issues()
