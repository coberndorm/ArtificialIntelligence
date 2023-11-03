import os
import subprocess
import sys
import shutil
# System imports
from venv import create
from os.path import join, expanduser, abspath
from subprocess import run

def create_and_activate_venv(venv_name):
    try:
        # Create virtual environment
        venv_dir = join(expanduser("."), venv_name)
        create(venv_dir, with_pip=True)
        print("Virtual environment created on: ", venv_dir)
        
        # Determine the appropriate Python executable based on the current platform
        if sys.platform == 'win32':
            python_executable = 'python'
        else:
            python_executable = 'python3'
        
        # Install packages in 'requirements.txt'
        run([python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        run([os.path.join(venv_dir, 'bin', 'pip'), 'install', '-r', abspath('requirements.txt')])
        
        print("Completed installation of requirements.")
    
    except Exception as e:
        raise Exception("Failed to create and activate the virtual environment: " + str(e))

def delete_venv(venv_name):
    # Determine the appropriate Python executable based on the current platform
    if sys.platform == 'win32':
        python_executable = 'python'
    else:
        python_executable = 'python3'
    
    try:
        subprocess.run(['deactivate'], shell=True, check=True)
    except subprocess.CalledProcessError:
        pass

    # Delete the virtual environment directory
    try:
        shutil.rmtree(venv_name)
        print(f"Virtual environment '{venv_name}' deleted successfully.")
    except FileNotFoundError:
        print(f"Virtual environment '{venv_name}' does not exist.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <dataset_path>")
        sys.exit(1)
    
    create_and_activate_venv(".venv")

    data_path = sys.argv[1]

    #TODO Statistichal analysis
    #TODO add preprocessing (normalization and hot encoding)
    #TODO clustering, different types
    #TODO save an html page with the results
    delete_venv(".venv")

    sys.exit(0)



