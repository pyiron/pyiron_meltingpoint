#!/bin/bash
mpiexec -n $1 lmp_mpi -in control.inp;
