#!/bin/sh
#PBS -l walltime=71:58:00
#PBS -l select=1:ncpus=128:mem=128GB:avx2=true

module load gromacs/2021.3-mpi gcc/9.3.0 mpi/intel-2019
# gromacs/2021.3-mpi: requires AVX2. Add avx2=true to #PBS resource selections

gmx_mpi mdrun -s topol.tpr > mdrun.log


