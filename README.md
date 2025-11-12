# PyTransport

PyTransport is a numerical tool for the computation of primordial correlators in inflationary cosmology. It provides a Python interface to C++ routines that handle computationally intensive tasks related to inflationary correlation functions.

## Features
- **Computes tree-level correlators** for multi-field models with canonical and non-canonical field space.
- **Integration of the Multi-Point Propagator (MPP) Approach** for enhanced numerical stability and accuracy.
- **Supports non-trivial field-space metrics**, allowing more flexibility in modeling.
- **Fast and efficient computations** leveraging a hybrid C++ and Python design.
- **Compatible with Python 3.x**, tested on Unix-like systems (Linux/macOS).

## Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager
- `virtualenv` (optional but recommended)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/PyTransport.git
   cd PyTransport
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Build and install PyTransport:
   ```sh
   python -m build
   pip install -e .
   ```

Alternatively, you can use the provided setup script:
```sh
python PyTransportSetup.py
```
This script will:
- Check for Python 3.8 or higher
- Upgrade pip
- Create and activate a virtual environment
- Install required dependencies
- Build and install PyTransport

## Project Configuration
PyTransport uses `pyproject.toml` for build configuration and package metadata. Key settings:
- **Build system:** `setuptools` and `wheel` are required.
- **Package structure:**
  - Includes `PyTransport`, `PyTransport.PyTransCpp`, and `PyTransport.PyTransPy`
  - Data files for `PyTransCpp` include C++ source files (`.cpp`, `.h`, `.hpp`)
  - Example scripts and templates are also packaged

### Package Metadata
- **Name:** PyTransport
- **Version:** 3.0.0
- **Authors:**
  - Andrea Costantini
  - David Mulryne
  - John Ronayne
- **License:** MIT
- **Supported Python versions:** `>=3.8`

## Usage

### Running PyTransport
After installation, activate the virtual environment and import PyTransport in Python:
```python
from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS
import PyTransDQ as PyT
```

## File Structure
- `PyTransScripts.py`: Python scripts that automate the editing, compiling, and execution of C++ code.
- `PyTransportSetup.py`: Setup script for installing and configuring PyTransport.
- `PyTrans.cpp`: Core C++ code implementing the numerical computations.
- `pyproject.toml`: Configuration file specifying build dependencies and metadata.
- `MANIFEST.in`: Specifies additional files to include in the package distribution.

## Licensing
PyTransport is distributed under the MIT License. See [LICENSE](LICENSE) for more details.

## Contributors
- Andrea Costantini
- David Mulryne
- John Ronayne

For any issues or contributions, feel free to submit a pull request or open an issue on GitHub.
