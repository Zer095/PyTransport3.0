from setuptools import setup, Extension
import os
from shutil import rmtree
import numpy

# Define the path to the 'include' directory that contains the header files
include_dir = [os.path.abspath(os.path.join(os.path.dirname(__file__), 'include'))]

# Set the CXXFLAGS environment variable to include the 'include' directory
os.environ["CXXFLAGS"] = f"-I{include_dir}"

# Define the extension module
PyTransLNC = Extension(
    "PyTransLNC",
    sources=["/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/PyTrans.cpp","/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/cppsrc/stepper/rkf45.cpp"],
    include_dirs = ['/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp', '/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/cppsrc/stepper', '/Users/apx050/Desktop/Projects/PyTransport/env/lib/python3.11/site-packages/numpy/core/include'],
    language="c++",
)

# Setup configuration
setup(
    name="PyTransLNC",
    ext_modules=[PyTransLNC],
    package_data={
        "PyTransLNC": [numpy.get_include()],
    },
)