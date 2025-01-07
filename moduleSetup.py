import os
import subprocess
import sys
import shutil
 
def main(name='PyTransport'):

    location = os.path.dirname(__file__)                            # Location where this file is located

    subprocess.run(['pip uninstall -y PyTransport'], shell=True)    
    print('------------------Build------------------------------')
    subprocess.run(['python -m build'], cwd=location, shell=True)   # Build the package
    print('------------------Install------------------------------')
    subprocess.run(['pip install -e .'], shell=True)                # Install the package

    shutil.rmtree(os.path.join(location, name+'.egg-info'))         # Remove unwanted folders
    # shutil.rmtree(os.path.join(location, 'build'))
    shutil.rmtree(os.path.join(location, 'dist'))


if __name__ == '__main__':
    main()