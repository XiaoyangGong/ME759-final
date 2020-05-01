#include "Stitcher.hpp"
#include <chrono>
#include <ratio>
#include <cmath>

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
	// constructor
Stitcher :: Stitcher(){
	minHessian = 400;
	detector = SURF::create(minHessian);
	matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
}

Mat Stitcher :: stitch(Mat& img1, Mat& img2){
	high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
	duration<double, std::milli> duration_sec;

	// detecting keypoints & computing descriptors
	start = high_resolution_clock::now();
	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
	end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Two img feature detection completed. Time taken: " << duration_sec.count() << endl;
	cout << "FOUND " << keypoints1.size() << " keypoints on first image" << endl;
	cout << "FOUND " << keypoints2.size() << " keypoints on second image" << endl;

    // Match descriptors with FLANN based matcher
    start = high_resolution_clock::now();
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    	// Filter matches using the Lowe's ratio test
    	// Can use OpenMP
	const float ratio_thresh = 0.75f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
            // TODO create match_score var from distance
		}
	}
	end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Img matching completed! Time taken: " << duration_sec.count() << endl;

    // Localize the object
	std::vector<Point2f> obj1;
	std::vector<Point2f> obj2;
	for( size_t i = 0; i < good_matches.size(); i++ ){
        // Get the keypoints from the good matches
		obj1.push_back(keypoints1[ good_matches[i].queryIdx ].pt );
		obj2.push_back(keypoints2[ good_matches[i].trainIdx ].pt );
	}

    // Find homography matrix
    // Note: order of obj2, obj1 does matter
    start = high_resolution_clock::now();
	H = cv::findHomography(obj2, obj1, RANSAC);
	end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Homography completed! Time taken: " << duration_sec.count() << endl;
    // Apply homography matrix and stitch
	start = high_resolution_clock::now();
	warpPerspective(img2, img_pano, H, Size(img2.cols + img1.cols, img2.rows));
	Mat half = img_pano(Rect(0, 0, img1.cols, img1.rows));
	img1.copyTo(half);
	end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Two image stitching completed! Time taken: " << duration_sec.count() << endl;
	return img_pano;
}
