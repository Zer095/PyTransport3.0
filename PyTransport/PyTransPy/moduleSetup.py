import os
import subprocess
import sys
import shutil

def setup(name):

    location = os.path.dirname(__file__)                            # Location where this file is located

    subprocess.run(['python -m build'], cwd=location, shell=True)   # Build the package
    subprocess.run(['pip install -v .'], shell=True)                # Install the package

    shutil.rmtree(os.path.join(location, name+'.egg-info'))         # Remove unwanted folders
    shutil.rmtree(os.path.join(location, 'build'))
    shutil.rmtree(os.path.join(location, 'dist')) 