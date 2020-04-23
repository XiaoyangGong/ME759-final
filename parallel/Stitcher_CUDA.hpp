#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/opencv_modules.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"


using namespace std;
using namespace cv;
using namespace cv::cuda;

class Stitcher{
private:
	double minHessian;
	SURF_CUDA surf;
	Ptr<cv::cuda::DescriptorMatcher> matcher;
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
	vector<DMatch> matches;
	GpuMat H;
	GpuMat img_pano;
public:
	Stitcher();
	GpuMat stitch(GpuMat& img1, GpuMat& img2);
};