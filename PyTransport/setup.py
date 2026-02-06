from setuptools import setup, Extension
import os
from shutil import rmtree
import numpy
import sys

# Define compiler flags based on the platform
if sys.platform.startswith('win'):
    # Flags for MSVC (Windows compiler)
    # /O2 is for maximum speed, /fp:fast enables fast floating-point math
    compile_args = ['/O2', '/fp:fast']
else:
    # Flags for GCC or Clang (Linux, macOS)
    # -O3 is the highest level of optimization
    # -march=native optimizes for the specific CPU architecture of your machine
    # -ffast-math allows for aggressive floating-point optimizations (use with care)
    compile_args = ['-O3', '-march=native', '-ffast-math']


# Define the path to the 'include' directory that contains the header files
include_dir = [os.path.abspath(os.path.join(os.path.dirname(__file__), 'include'))]

# Set the CXXFLAGS environment variable to include the 'include' directory
os.environ["CXXFLAGS"] = f"-I{include_dir}"

# Define the extension module
PyTransSFA = Extension(
    "PyTransSFA",
    sources=[r"/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/PyTrans.cpp",r"/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/cppsrc/stepper/rkf45.cpp"],
    include_dirs = ['/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp', '/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/cppsrc/stepper', '/Users/apx050/Desktop/Projects/PyTransport3.0/venv/lib/python3.12/site-packages/numpy/_core/include'],
    #extra_compile_args=compile_args,
    language="c++",
)

# Setup configuration
setup(
    name="PyTransSFA",
    ext_modules=[PyTransSFA],
    package_data={
        "PyTransSFA": [numpy.get_include()],
    },
)