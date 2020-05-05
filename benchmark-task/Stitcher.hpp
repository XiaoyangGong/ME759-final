#include <iostream>
#include <vector>
#include <string>
#include "opencv2/core.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class Stitcher{
private:
	double minHessian;
	Ptr<SURF> detector;
	Ptr<DescriptorMatcher> matcher;
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	std::vector<std::vector<DMatch>> knn_matches;
	Mat H;
	Mat img_pano;
public:
	Stitcher();
	Mat stitch(Mat& img1, Mat& img2);
};