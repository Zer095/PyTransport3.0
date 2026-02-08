# Installation

## Prerequisites
* Python 3.8 or higher
* `gcc` or `g++` compiler

## Setup
1. Clone the repository:
   ```sh
   git clone [https://github.com/your-repo/PyTransport.git](https://github.com/your-repo/PyTransport.git)
   cd PyTransport
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
