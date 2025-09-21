#!/bin/bash
#PJM -L rscgrp=cx-small
#PJM -L node=1
#PJM -L elapse=00:10:00
#PJM --mpi proc=1
#PJM -j
#PJM -S

module load gcc/11.3.0
module load hpc_sdk/23.1

echo "====== v1.2.0 execution ======"
date
hostname
nvidia-smi

./gemm_cuda_v1.2.0

echo "====== End ======"
date
