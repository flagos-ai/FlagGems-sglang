#!/bin/bash
yhrun -p h100x -G 1 pytest test_axpy.py::test_accuracy_axpy_real -s -v
#yhrun -p h100x -G 1 pytest test_axpy.py
