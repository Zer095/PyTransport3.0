# Usage Tutorial

Using PyTransport involves two main phases: defining your cosmological model (Potential and Metric) and running the numerical evolution.

## Phase 1: Model Setup

You must define your model using **SymPy** syntax. This allows PyTransport to automatically calculate the necessary derivatives for the transport equations.

Create a setup file (e.g., `my_model_setup.py`):

```python
import sympy as sym
from PyTransport.PyTransPy import PyTransSetup

# 1. Define the number of fields and parameters
nF = 2 
nP = 2 
f = sym.symarray('f', nF)
p = sym.symarray('p', nP)

# 2. Define the potential (V) using SymPy notation
# Example: V = m1*phi + m2*chi
V = p[0]*f[0] + p[1]*f[1] 

# 3. Process the potential and compile the C++ backend
PyTransSetup.potential(V, nF, nP, simplify=True)
PyTransSetup.compileName("MyModel")