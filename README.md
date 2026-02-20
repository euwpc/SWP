# SWP – Swedish Weather Project

Visualize meteorological data using high-quality maps generated from NetCDF files.  
Built with Python, Matplotlib, Cartopy, NumPy, and SciPy.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python version">
  <img src="https://img.shields.io/badge/Dependencies-matplotlib%20%7C%20cartopy%20%7C%20netCDF4-success?style=for-the-badge" alt="Dependencies">
  <img src="https://img.shields.io/github/repo-size/euwpc/SWP?style=for-the-badge&color=informational" alt="Repo size">
  <img src="https://img.shields.io/github/last-commit/euwpc/SWP?style=for-the-badge&color=important" alt="Last commit">
</p>

---

## Features

- High-resolution geographic mapping with Cartopy  
- Support for contour plots, filled contours, wind barbs, quiver vectors, and more  
- Compatible with common meteorological variables (temperature, pressure, wind, precipitation, etc.)  
- Efficient handling of large NetCDF datasets  
- Simple command-line execution  

---

## Quick Start

### 1. Install Visual Studio Code (recommended)

Download from: https://code.visualstudio.com/

### 2. Clone the repository

In VS Code:  
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)  
- Select **Git: Clone** if it doesn't show, write **Git: Clone** yourself. **YOU NEED TO SIGN IN WITH YOUR GITHUB ACCOUNT!**
- Enter the URL:  
  `https://github.com/euwpc/SWP.git`  
- Choose a destination folder and confirm  

### 3. Trust the folder authors

When prompted with “Do you trust the authors of the files in this folder?”  
select **Yes, I trust the authors**

### 4. Install required Python packages

Open the integrated terminal (`Terminal → New Terminal` or `` Ctrl+` ``) and run:


pip install requests matplotlib cartopy numpy scipy pillow netCDF4

On some Windows systems, use:

py -m pip install requests matplotlib cartopy numpy scipy pillow netCDF4


### 5. Run the program
In the terminal, execute:

python main.py

or on Windows (if needed):

py main.py

The first run may take several minutes while map data is downloaded and cached.
Output graphics will appear in the project folder.

