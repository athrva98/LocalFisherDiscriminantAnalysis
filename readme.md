# Local Fisher Discriminant Analysis (LFDA) Implementation

This repository contains the implementation of Local Fisher Discriminant Analysis (LFDA), a supervised dimensionality reduction technique. LFDA is particularly useful in machine learning and pattern recognition for maintaining local class discriminability while maximizing class separability. This implementation is modular, consisting of the core LFDA algorithm, metrics calculations, and Python bindings for easy integration into data science workflows.

## Overview

LFDA aims to find a transformation that maximizes the between-class scatter while minimizing the within-class scatter, taking into account the local structure of the data. This implementation provides a flexible, efficient approach to LFDA, including support for various distance metrics and embedding techniques.

### Contents

- `lfda.h` & `.cpp`: Core LFDA algorithm and class definition.
- `metrics.h` & `.cpp`: Distance metrics calculations.
- Python bindings: Nanobind usage to expose LFDA functionality to Python.

## Features

- **LFDA Algorithm**: Implements the LFDA algorithm for supervised dimensionality reduction, including fitting the model to input data and transforming new data.
- **Distance Metrics**: Supports multiple distance metrics, including Euclidean, Manhattan, cosine, and more, for versatile pairwise distance computations.
- **Python Integration**: Provides Python bindings using Nanobind, making the LFDA implementation accessible in Python environments.
- One improvement over this implementation is Kernel LFDA, which willbe implemented soon.

## Getting Started

### Prerequisites

- Eigen: A high-level C++ library for linear algebra operations.
- OpenMP: For parallel processing and improving computational efficiency.
- Nanobind: For creating Python bindings.

### Installation

1. **Clone the repository**:
   
   ```sh
   git clone --recursive-submodules https://github.com/athrva98/LocalFisherDiscriminantAnalysis.git
   ```

2. **Compile the C++ Code**:
* Ensure Eigen and OpenMP are installed and configured in your environment.

* Ensure that you have python installed in your environment. To check, open `cmd` and run

* ```
  python --version
  ```
  
  This project assumes python 3.8, but you can build the package for python 3.8+ versions. To do this, make appropriate changes in the `CMakeLists.txt` file.

* Use your preferred C++ compiler to compile the source files, adjusting include paths as necessary.

* For windows users, using visual studio, do the following

* ```
  # in the root directory of the project
  mkdir build
  cd build
  cmake ..
  cmake build .
  ```

* The above should generate a folder called `build` containing the Visual Studio Solution. Then open the solution file and build the `ALL_BUILD` target. Make sure to build for Release x64 configuration.

* You should see the folder `Release` created. Inside this folder, you will find `lfda.pyd`

* Move the `lfda.pyd` file into the `python_lib_install` folder and follow the readme contained in the folder.

* After this, you should have the package `lfda` installed.

* You can test the package by following the readme in `python_test` folder.
  
  ### C++ Usage
  
  If you would like to use the package in a C++ project, follow the usage: 
  
  ```C++
  #include "lfda.h"
  
  // Initialize LFDA with desired parameters
  LFDA lfda(n_components, k, EMBEDDING_TYPE::WEIGHTED);
  
  // Fit the model
  lfda.Fit(X, y);
  
  // Transform new data
  Eigen::MatrixXd transformed = lfda.Transform(new_X);
  ```
  
  ### Python Usage
  
  To use the package as a python module, follow the pattern in `python_test\lfda_test.py` or
  
  ```python
  from lfda import lfda
  
  # Initialize and fit LFDA
  model = lfda.LFDA(n_components=2, k=5, embedding=lfda.EMBEDDING_TYPE.WEIGHTED)
  model.fit(X, y)
  
  # Transform new data
  transformed = model.transform(new_X)
  
  ```
  
  ### Contributing
  
  Contributions to improve or extend the functionality of this LFDA implementation are welcome. Please follow the standard pull request process:
  
  1. Fork the repository.
  2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
  3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
  4. Push to the branch (`git push origin feature/AmazingFeature`).
  5. Open a pull request.

License
-------

Distributed under the MIT License. See `LICENSE` for more information.
Acknowledgments
---------------

* The Eigen library for linear algebra operations.
* OpenMP for parallel computation capabilities.
* Nanobind for facilitating Python integration.
