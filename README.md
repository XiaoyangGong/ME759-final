# ME759-final

Compile:
g++ -o feature feature.cpp `pkg-config opencv --cflags --libs`

g++ -o ref ref.cpp `pkg-config opencv --cflags --libs`
For CUDA-OpenCV:
Dependencies:
module load gcc/0_cuda/7.1.0
module load cuda/10.0

