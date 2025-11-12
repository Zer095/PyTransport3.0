.. _api_reference:

API Reference
=============

This section provides a detailed reference for both the Python utility scripts and the core C++ extension functions.

Python Utility Scripts (`PyTransScripts.py`)
--------------------------------------------

These functions provide helpful utilities for setting up initial conditions and running the numerical solvers.

.. automodule:: PyTransport.PyTransPy.PyTransScripts
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: unPackAlp, unPackSig

Core C++ Functions (`PyTrans.cpp`)
----------------------------------

These are the core numerical functions compiled from C++ that perform the heavy lifting for the transport calculations. The documentation is generated directly from the C++ source code comments.

.. doxygenfile:: PyTrans.cpp
   :project: PyTransport3.0
