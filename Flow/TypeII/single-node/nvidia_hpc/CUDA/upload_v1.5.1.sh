#!/bin/bash
# Upload script for v1.5.1

echo "Uploading gemm_cuda_v1.5.1.cu to Flow..."

# Create SFTP batch file
cat > /tmp/sftp_batch_v151.txt << EOF
cd /data/group1/z42448z/VibeCodeHPC/GEMM/v0.6.10/multi/ex1/
put gemm_cuda_v1.5.1.cu
EOF

# Execute SFTP
sftp -b /tmp/sftp_batch_v151.txt z42448z@flow.cc.nagoya-u.ac.jp

# Clean up
rm /tmp/sftp_batch_v151.txt

echo "Upload complete"