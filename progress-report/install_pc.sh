echo "Start OpenCV installation..."
echo "Warning: You need to have CUDA 10.0 installed"
echo "Updating system..."
sudo apt update
sudo apt upgrade

echo "Installing generic tools..."
sudo apt install build-essential cmake pkg-config unzip yasm git gcc-6 checkinstall

echo "Installing Image I/O libs..."
sudo apt install libjpeg-dev libpng-dev libtiff-dev

echo "Installing Video/Audio Libs libs..."
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev

echo "Installing Adaptive Multi Rate Narrow Band (AMRNB) and Wide Band (AMRWB) speech codec..."
sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev

echo "Installing Cameras programming interface libs..."
sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~

echo "GTK lib for the graphical user functionalites..."
sudo apt-get install libgtk-3-dev

echo "Installing Parallelism library C++ for CPU..."
sudo apt-get install libtbb-dev

echo "Installing Optimization libraries for OpenCV..."
sudo apt-get install libatlas-base-dev gfortran

echo "Installing optional libs..."
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

echo "Pulling OpenCV and OpenCV-contrib from Git repo..."
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && git checkout 3.4.1 && cd ..
cd opencv_contrib && git checkout 3.4.1 && cd ..

echo "Procced with the installation.."
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_C_COMPILER=/usr/bin/gcc-6 \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \ ..

make -j($nproc)
sudo make install

echo "Include libs in env"
sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig



