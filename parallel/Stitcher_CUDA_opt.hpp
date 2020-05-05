#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/opencv_modules.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"


using namespace std;
using namespace cv;
using namespace cv::cuda;

class Stitcher_CUDA_opt{
private:
	SURF_CUDA surf;
	Ptr<cv::DescriptorMatcher> matcher;
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
	vector<vector<cv::DMatch> > knn_matches;
	Mat H;
	GpuMat img_pano;
	
public:
	Stitcher_CUDA_opt();
	GpuMat stitch(GpuMat& img1, GpuMat& img2);
};