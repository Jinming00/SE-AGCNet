# PyAGC - Time-Frequency Automatic Gain Control

## Introduction

This code is derived from [@jorgehatccrma/pyagc](https://github.com/jorgehatccrma/pyagc.git), which is a Python 2 implementation based on the original Matlab code by D. Ellis (2010), "Time-frequency automatic gain control", web resource, available: http://labrosa.ee.columbia.edu/matlab/tf_agc/.

We have modified the code to be compatible with Python 3. Additionally, through extensive experimental validation, we have optimized two key parameters to achieve better Speech Enhancement (SE) and Automatic Speech Recognition (ASR) performance:

- **t_scale = 1.0** (temporal scale parameter)
- **f_scale = 8.0** (frequency scale parameter)


## References

D. Ellis (2010), "Time-frequency automatic gain control", web resource, available: http://labrosa.ee.columbia.edu/matlab/tf_agc/

Original Python 2 implementation: https://github.com/jorgehatccrma/pyagc.git
