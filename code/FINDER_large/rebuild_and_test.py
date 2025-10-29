#!/usr/bin/env python
"""
Script to rebuild the Cython extensions and test the fixes
"""

import subprocess
import sys
import os

def rebuild_extensions():
    """Rebuild the Cython extensions"""
    print("Rebuilding Cython extensions...")
    try:
        result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        else:
            print("Build successful!")
            return True
    except Exception as e:
        print(f"Error during build: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("Testing basic functionality...")
    try:
        # Import the modules
        import GraphDQN
        import mvc_env
        import nstep_replay_mem
        
        print("All modules imported successfully!")
        
        # Test creating a simple environment
        env = mvc_env.py_MvcEnv(100.0)
        print("MvcEnv created successfully!")
        
        # Test creating replay memory
        replay_mem = nstep_replay_mem.py_NStepReplayMem(1000, 0.99)
        print("NStepReplayMem created successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Rebuilding and Testing GraphDQN ===")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Rebuild
    if not rebuild_extensions():
        print("Build failed, exiting...")
        sys.exit(1)
    
    # Test
    if not test_basic_functionality():
        print("Testing failed, but build was successful")
        sys.exit(1)
    
    print("=== 