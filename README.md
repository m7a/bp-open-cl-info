---
section: 11
x-masysma-name: ma_open_cl_info
title: Ma_Sys.ma OpenCL info and testing program
date: 2020/02/17 12:43:43
lang: en-US
author: ["Linux-Fan, Ma_Sys.ma (Ma_Sys.ma@web.de)"]
keywords: ["opencl", "compute", "c"]
x-masysma-version: 1.0.0
x-masysma-repository: https://www.github.com/m7a/bp-open-cl-info
x-masysma-website: https://masysma.lima-city.de/11/ma_open_cl_info.xhtml
x-masysma-owned: 1
x-masysma-copyright: |
  Copyright (c) 2016, 2020 Ma_Sys.ma.
  For further info send an e-mail to Ma_Sys.ma@web.de.
---
Name
====

`ma_open_cl_info` -- Ma_Sys.ma OpenCL info and testing program

Synopsis
========

	ma_open_cl_info [-h|--help] [-n] [-g] [-p <N>]

Description
===========

Allows the user to test the avaiability of OpenCL computation with GPUs and
the CPU. The program is intended to give a rough estimate on the speed.
Note that CPU and GPU timings are not comparable due to differences in
implementation (see below for the details).

## Algorithm

This program multiplies two random float/single-matrices A and B.
A and B are nxn-matrices where n is the “problem size”.
In order to caluclate the result, the OpenCL device needs to provide three
times the memory necessary to hold one matrix (A, B and result = 3 Matrices)

To be able to compute a comparison-result on CPU we not simply calculate
A $\times$ B, but instead (A $\times$ B)^64^. This way we can utilize
exponentiation by squaring on the CPU. In order to run longer on the GPU
(for more reliable measures), we do _not_ use exponentiation by squaring on
the OpenCL device.

Options
=======

--  -----------------------------------------------------------
-h  Displays help screen.
-n  Disables result verification (no CPU precalculation).
-g  Requires OpenCL to be run on GPUs only. (Does not imply -n)
-p  Configure problem size to be N, default 2048
--  -----------------------------------------------------------

Examples
========

Typical invocation on a laptop with Intel GPU and NVidia GPU available:

~~~
$ ./ma_open_cl_info -p 1024
Ma_Sys.ma OpenCL info and testing program 1.0.0, Copyright (c) 2016 Ma_Sys.ma.
[...]

Info: Changed problem size to 1024
System information
Platform 0: 1 devices.
  Device 0:
    Vendor:        Intel
    Name:          Intel(R) HD Graphics Haswell Ultrabook GT3 Mobile
    Compute Units: 40
    Global Memory: 2048 MiB
    Local Memory:  64 KiB
Platform 1: 1 devices.
  Device 0:
    Vendor:        NVIDIA
    Name:          NVD7
    Compute Units: 2
    Global Memory: 1048576 MiB
    Local Memory:  48 KiB

Tests
Initializing Tests (asz=4 MiB) ... talloc=0.0 trnd=0.0 tcalc=14.0 tS=14.0
Platform 0
  Device 0
    Comparing results... finished
    tinit=0.0, tmem=0.0, tcalc=32.0, tmem2=0.0, tcmp=0.0, tS/OCL=32.0, tS=32.0
Platform 1
  Device 0
Failed to run OpenCL statement, line 530, error code -11 / CL_BUILD_PROGRAM_FAILURE.
Failed to compile program: invalid source
~~~

Note that despite the fact that the NVIDIA GPU is reported, it fails to execute
the  OpenCL (the error message does not seem to be overly helpful). This is
likely due to some missing parts of the proprietary NVIDIA driver which might
be necessary to utilize that GPU. In the example, the `tS` times are relevant.
Here, they were 14.0 sec for the initialization and 32.0 sec for the actual
computation.

Compilation
===========

Compile the tool with `ant` or build the package with `ant package`.
Dependencies for building OpenCL applications are often hard to satisfy.
Check `bdep` in `build.xml` for a list of Debian packages which seemed to be
sufficient for compilation at the time of this writing.

Bugs
====

 * May fail to compile if sizes of `float` are different than expected.
   This problem is known to occur on Windows 8.1 and Windows 10.
 * May fail to execute if the work size (`asz`) is greater than 100 MiB because
   of timeouts. This is a limitation of the current software design (which does
   not allow the computation on the GPU to be temporarily suspended).
