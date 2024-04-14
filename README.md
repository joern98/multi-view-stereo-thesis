# Combination of two stereo depth cameras to improve depth quality in the distance

### Code zur Bachelorarbeit
> Jörn Eggersglüß  
> Studiengang Mediendesigninformatik  
> Fakultät 4, Abteilung Informatik  
> Hochschule Hannover  

## Preface
This repository contains the code for the thesis,
refactored and restructured for readability based on the development repositories
https://github.com/joern98/wide-stereo and https://github.com/joern98/device_utility.

## Calibration
Camera calibration is performed using `utility/camera_calibration.py`.
The calibration result is saved as a `npy` file for later use.
Additionally, a human-readable `json` file is saved.

## Capture
Static samples are captured using `utility/capture.py`
A calibration `.npy` file must be given as input to the script.

## First approach
Implementation of the first approach in `first_approach.py`.
The script takes the path to a capture directory as produced by `capture.py`.
It shows the results of the algorithm,
which may then be saved to a subdirectory of the capture directory.

## Second approach / plane sweep approach
Implementation of the second approach in `plane_sweep.py`.
The script takes the path to a capture directory as produced by `capture.py`.

Furthermore, `plane_sweep_impl.py` contains the code for
the consistency computation as a Cython extension.
It can be compiled by running
```shell
$ python setup.py build_ext --inplace
```
The built `.pyd` module is then imported in `plane_sweep.py` with the line 
```python
from plane_sweep_ext import compute_consistency_image
```
Since the algorithm is quite slow, after completion,
the user is asked if the cost-volume should be saved.

The output of this is the same as with the first approach,
depth map is shown to the user and may be saved to a subdirectory of the capture directory  

## Samples
The `samples` directory contains the sample scenes mentioned in the thesis.