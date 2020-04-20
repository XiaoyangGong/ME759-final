echo "Start OpenCV installation on Euler..."

echo "Pulling OpenCV and OpenCV-contrib from Git repo..."
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && git checkout 3.4.1 && cd ..
cd opencv_contrib && git checkout 3.4.1 && cd ..

echo "Procced with the installation.."
echo "Default local lib directory: ~/lib"
echo "You can change local lib location by modifying this script"
build_path="~/lib"
cd opencv && mkdir build && cd build
module load cuda/10.0
module load gcc/0_cuda/7.1.0
module load gcc/6.1.0 
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_C_COMPILER=/usr/local/gcc/6.1.0/bin/gcc \
-D CMAKE_INSTALL_PREFIX=$build_path \
-D WITH_TBB=ON \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_QT=OFF \
-D WITH_GSTREAMER=OFF \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..

make -j($nproc)
sudo make install

echo "Include lib path and linker path in .bashrc"

export PKG_CONFIG_PATH=$build_path/lib64/pkgconfig/:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$build_path/lib64:LD_LIBRARY_PATH


