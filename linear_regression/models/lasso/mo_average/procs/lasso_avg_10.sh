#!/bin/bash -l
#PBS -N grid_mo_lasso_avg_7_7_np_10
#PBS -A P48500028
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=50GB
#PBS -l walltime=20:00:00
#PBS -q casper
#PBS -j oe

###module swap
source /glade/u/home/rossamower/work/miniconda3/etc/profile.d/conda.sh
conda activate ucb_w207

### Set TMPDIR as recommended
export TMPDIR=/glade/scratch/$USER/temp_serial
mkdir -p $TMPDIR

PROC="10"


###module swap
time python grid_mo_lasso_avg.py ${PROC}

