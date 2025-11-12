import os
import subprocess
import sys
import shutil
 
def installPyTransport(name='PyTransport'):
    # Get the directory of the current script
    location = os.path.dirname(__file__)

    # Uninstall any existing version of PyTransport
    subprocess.run(['pip', 'uninstall', '-y', 'PyTransport'], shell=True)    
    print('------------------Build PyTransport------------------------------')
    
    # Build the PyTransport package in the current directory
    subprocess.run(['python -m build'], cwd=location, shell=True)   
    print('------------------Install PyTransport------------------------------')
    
    # Install the package in editable mode
    subprocess.run(['pip install -e .'], shell=True)                

    # Remove unnecessary build-related directories and files
    shutil.rmtree(os.path.join(location, name+'.egg-info'))         
    shutil.rmtree(os.path.join(location, 'dist'))
    os.remove(os.path.join(location, 'setup.py'))

def main():
    # Check Python version
    print('------------------Check Python------------------')
    if int(sys.version[0]) != 3:
        print('Your Python version is not upgraded to Python 3.\n Please upgrade Python before installing PyTransport')
        exit()
    else:
        print(sys.version)

    # Upgrade pip to the latest version
    print('------------------Upgrade Pip------------------')
    subprocess.run(['pip', 'install', '--upgrade', 'pip'], shell=True) 

    # Create a virtual environment
    print('------------------Create venv------------------')
    subprocess.run(['python -m venv venv'], shell=True) 

    # Activate the virtual environment (Linux/Mac only; Windows needs different approach)
    print('------------------Activate Venv------------------')
    subprocess.run(['chmod +x activate.sh'], shell=True)  # Ensure activation script is executable
    subprocess.run(['source activate.sh'], shell=True)    # Activate the environment

    # Install required dependencies from requirements.txt
    print('------------------Install Package------------------')
    subprocess.run(['pip install -r requirements.txt'], shell=True) 

    # Install the PyTransport package
    installPyTransport()
    
    print('To use PyTransport, you must activate the virtual environment.\nTo activate the virtual environment, run: source venv/bin/activate')

if __name__ == '__main__':
    main()