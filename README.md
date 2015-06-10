# python_igraph_numpy_c_ext

#### This code is largely untested and I am sharing it for feedback purposes. It probably contains errors that result in memory leaks and incorrect results. Use at your own risk!

Rough but working C extension to Python that uses iGraph and Numpy libraries in both Python and C

##### Requires a very recent install of python-igraph >= Version: 0.7.1.post4

Code hows a way to use C-iGraph functions on a graph created in Python and 
how to return computation results as both floats and numpy arrays. 
The numpy arrays are created in Python and filled by the C code. 
It is rough code, pieced together from many sources, and the sample 
computations are trivial examples.

I would welcome feedback that improves code quality or implements better practices. 


##### How to use:
Download the files to a directory. If iGraph headers are not in '/usr/local/include' edit below

In terminal:

    % cd <install directory>
    % python setup.py build_ext --inplace -I/usr/local/include
    % python test_ignp.py 
