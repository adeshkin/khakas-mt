# khakas-mt
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install wheel
pip install ninja
export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn --no-build-isolation