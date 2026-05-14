#!/bin/bash
yhrun -p h100x -G 1 pytest test_scal.py::test_accuracy_scal_real -s -v
#yhrun -p h100x -G 1 pytest test_axpy.py
