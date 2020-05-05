#include <string>
#include <iostream>
#include <vector>
#include<time.h> 

#include "Stitcher.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace std;
using namespace cv;



int main() {
	Mat img1 = cv::imread("../images/bryce_left_01.png", IMREAD_GRAYSCALE);
	CV_Assert(!img1.empty());

    Mat img2 = cv::imread("../images/scottsdale_left_01.png", IMREAD_GRAYSCALE);
    CV_Assert(!img2.empty());
    
    Mat img3 = cv::imread("../images/scottsdale_right_01.png", IMREAD_GRAYSCALE);
    CV_Assert(!img3.empty());

    Mat img4 = cv::imread("../images/sedona_left_01.png", IMREAD_GRAYSCALE);
    CV_Assert(!img4.empty());
    Mat big1;
    Mat big2;

    hconcat(img1, img2, big1);
    hconcat(img3, img4, big2);


    imshow("big1", big1);
    imshow("big2", big2);

    //float homography_data[9] = {1, 0, (float)(rand() % mcat.cols), 0, 1, (float)(rand() % mcat.rows), 0, 0, 1};
    // {(enlarge x direction), (rotate: neg clk-wise, pos cclk-wise), x_offset;
	//  (like push up/down), (enlarge y direction), y_offset;
	//	}
    //float homography_data[9] = {1, 0.75, -256, 0.05, 1, -128, 0, 0, 1};
    /*
    float homography_data[9] = {1, 0.75, -256, 0.05, 1, -128, 0, 0, 1};
    Mat H = Mat(3, 3, CV_32F, homography_data);
    Mat tran;
    cv::warpPerspective(origin, tran, H, Size(tran.cols, tran.rows));
    //imshow("origin", origin);
    imshow("xform", tran);
    waitKey(0);
    */

    
    Stitcher* st = new Stitcher();
    Mat pano;
    pano = st->stitch(big1, big2);
    imshow("Pano", pano);
    waitKey(0);
    
}