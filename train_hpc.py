#!/usr/bin/python3
import subprocess
import os
import sys

# Global base directory where the repository lives on the HPC
BASE_DIR = '/home/hutlab_int/Hegde_netravaad'

def get_home_directory():
    home_directory = os.path.expanduser("~")
    return home_directory

def create_python_env_and_run_script():
    env_name = 'hegde_netravaad'  
    # Create the environment directly in the base directory
    envpath = os.path.join(BASE_DIR, env_name)

    print(f"==========================================")
    print(f"Running HPC Training Job globally")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Environment:    {envpath}")
    print(f"==========================================")

    # Check if the virtual environment already exists
    if not os.path.exists(envpath):
        print(f"=> Virtual environment doesn't exist. Creating {envpath}...")
        subprocess.run([sys.executable, '-m', 'venv', envpath])
    else:
        print(f"=> Virtual environment {envpath} already exists.")

    print("=> Env ready!")
    
    # Paths to pip and python inside the virtual environment
    pip_exe = os.path.join(envpath, 'bin', 'pip')
    python_exe = os.path.join(envpath, 'bin', 'python3')
    
    # --------------------------------------- Package installation ------------------------------------
    print("\n=> Upgrading pip...")
    subprocess.run([pip_exe, 'install', '--upgrade', 'pip'])
    
    print("\n=> Installing PyTorch with CUDA explicitly...")
    subprocess.run([pip_exe, 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu118'])

    print("\n=> Installing ultralytics dependencies (editable mode)...")
    ultralytics_path = os.path.join(BASE_DIR, 'ultralytics')
    subprocess.run([pip_exe, 'install', '-e', ultralytics_path])
    
    # --------------------------------------- Run Training Script -------------------------------------
    print("\n=> Starting YOLO training...")
    
    # We must pass the correct absolute PYTHONPATH 
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = ultralytics_path
    
    # Command to run YOLO training with absolute paths
    train_script = os.path.join(BASE_DIR, 'train_yolo.py')
    model_cfg = os.path.join(BASE_DIR, 'ultralytics/ultralytics/cfg/models/11/eyewave_transformer.yaml')
    data_yaml = os.path.join(BASE_DIR, 'nano_subset-Copy/data.yaml')
    
    run_script_cmd = [
        python_exe,
        train_script,
        '--data', data_yaml,
        '--model', model_cfg,
        '--epochs', '50',
        '--batch', '32',
        '--imgsz', '224'
    ]
    
    # Setting cwd explicitly so any relative paths created by ultralytics (like runs/detect) 
    # correctly output into the Hegde_netravaad folder regardless of where the script was called from.
    subprocess.run(run_script_cmd, env=env_vars, cwd=BASE_DIR)
    print("\n=> Training script completed.")

if __name__ == '__main__':
    create_python_env_and_run_script()
