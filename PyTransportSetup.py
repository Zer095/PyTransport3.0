import os
import subprocess
import sys
import shutil
 
def installPyTransport(name='PyTransport'):

    location = os.path.dirname(__file__)                            # Location where this file is located

    subprocess.run(['pip uninstall -y PyTransport'], shell=True)    
    print('------------------Build PyTransport------------------------------')
    subprocess.run(['python -m build'], cwd=location, shell=True)   # Build the package
    print('------------------Install PyTransport------------------------------')
    subprocess.run(['pip install -e .'], shell=True)                # Install the package

    shutil.rmtree(os.path.join(location, name+'.egg-info'))         # Remove unwanted folders
    shutil.rmtree(os.path.join(location, 'dist'))
    os.remove(os.path.join(location, 'setup.py'))

def main():

    # Check Python version
    print('------------------Check Python------------------')
    if int(sys.version[0]) != 3:
        print('Your Python version is not upgraded to Python 3.\n Pleas upload Python before installing PyTransport')
        exit()
    else:
        print(sys.version)

    # Upgrade pip
    print('------------------Upgrade Pip------------------')
    subprocess.run(['pip install --upgrade pip'], shell=True) 

    # Create virtual environment
    print('------------------Create venv------------------')
    subprocess.run(['python -m venv venv'], shell=True) 

    # Activate virtual environment
    print('------------------Activate Venv------------------')
    subprocess.run(['chmod +x activate.sh'], shell=True)
    subprocess.run(['source activate.sh'], shell=True)

    # Installed required package
    print('------------------Install Package------------------')
    subprocess.run(['pip install -r requirements.txt'], shell=True) 

    # Install PyTransport
    installPyTransport()

    print('To use PyTransport, you must activate the virtual environment.\n To activate the virtual environment run: source venv/bin/activate')

if __name__ == '__main__':
    main()