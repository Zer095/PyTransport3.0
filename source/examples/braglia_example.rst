Braglia Example
===============

The Braglia example demonstrates the use of PyTransport for analyzing a specific inflationary model. This example includes scripts for setting up the model, running background evolution, calculating 2-point and 3-point correlation functions, and plotting the results.

Key scripts and their purposes:

* :ref:`LNCcmb.py <braglia-lnccmb>`: A primary script that orchestrates the background evolution, 2-point, and 3-point calculations for the Braglia model. It also handles plotting the output.
* :ref:`Lsetup.py`: Used for setting up the specific model parameters and compiling the C++ backend for the Braglia model.
* :ref:`Lspectra.py`: Likely contains logic for calculating spectra related to this model.
* :ref:`NBplots.py`: Scripts for generating specific plots for the Braglia model results.
* :ref:`Spectra_plots.py`: Additional plotting scripts for spectral data.
* :ref:`TestNB.py`: A test script for numerical background calculations.
* :ref:`comb.py`: (Purpose needs to be inferred or described if known, e.g., combining data.)
* :ref:`example.py`: A general example usage script.

.. _braglia-lnccmb:

LNCcmb.py
---------
This script performs a full run of the LNC (Large Non-Canonical) model, including background evolution, 2-point and 3-point correlation function computations using both PyTransport's native solver and the MPP (Mean Particle Production) method, and generates comparison plots.

.. literalinclude:: ../../PyTransport/Examples/Braglia/LNCcmb.py
   :language: python
   :linenos:
   :caption: PyTransport/Examples/Braglia/LNCcmb.py
