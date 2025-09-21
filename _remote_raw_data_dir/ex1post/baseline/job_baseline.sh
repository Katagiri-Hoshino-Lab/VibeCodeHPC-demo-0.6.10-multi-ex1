#!/bin/bash
#PJM -L "rscunit=cx"
#PJM -L "rscgrp=cx-small"
#PJM -L "node=1"
#PJM -L "elapse=00:10:00"
#PJM -j
#PJM -o job_baseline.out

module load gcc/11.3.0

echo "Job started at: $(date)"
echo "Running baseline GEMM..."
./gemm_base
echo "Job ended at: $(date)"
