Installation
============

To install PyTransport, follow these steps:

1.  **Prerequisites:**
    * Python 3.x
    * `numpy`
    * `scipy`
    * `matplotlib`
    * `sympy`
    * `gravipy` (if you intend to use field metric calculations)
    * `mpi4py` (if you intend to use MPI parallelization)

    You can install most Python dependencies using pip:
    
    .. code-block:: bash

        pip install numpy scipy matplotlib sympy mpi4py gravipy

2.  **Clone the repository:**

    .. code-block:: bash

        git clone https://github.com/your-repo/PyTransport3.0.git
        cd PyTransport3.0

3.  **Install PyTransport:**

    PyTransport contains C++ components that need to be compiled. Use the `PyTransportSetup.py` script:

    .. code-block:: bash

        python PyTransportSetup.py compile_transport <module_name>

    Replace `<module_name>` with the desired name for the compiled module (e.g., `MyPyTrans`). This will compile the C++ backend and make the module available.

    For example:
    
    .. code-block:: bash

        python PyTransportSetup.py compile_transport PyTransLNC

    Refer to the `PyTransportSetup.py` script for more details on compilation options.

4.  **Verify Installation:**
    You can verify the installation by running some of the examples provided in the `PyTransport/Examples` directory.
