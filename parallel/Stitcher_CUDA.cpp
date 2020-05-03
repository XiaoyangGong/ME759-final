#include "Stitcher_CUDA.hpp"

#include <chrono>
#include <ratio>
#include <cmath>
using std::chrono::high_resolution_clock;
using std::chrono::duration;

using namespace std;
using namespace cv;
using namespace cv::cuda;
	// constructor
Stitcher_CUDA :: Stitcher_CUDA(){
	matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
}
// inline
GpuMat Stitcher_CUDA :: stitch(GpuMat& img1, GpuMat& img2){
	high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
	duration<double, std::milli> duration_sec;

	// detecting keypoints & computing descriptors
	start = high_resolution_clock::now();
    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);
	end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Two img feature detection completed. Time taken: " << duration_sec.count() << endl;

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    
    // Match descriptors with FLANN based matcher
    start = high_resolution_clock::now();
    matcher->knnMatch(descriptors1GPU, descriptors2GPU, knn_matches, 2);
    end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Img matching completed. Time taken: " << duration_sec.count() << endl;
	std::cout << "knn_matches=" << knn_matches.size() << std::endl;


    // Filter matches using the Lowe's ratio test
    // Can use OpenMP
	const float ratio_thresh = 0.7f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
            // TODO create match_score var from distance
		}
	}
	std::cout << "Good matches =" << good_matches.size() << std::endl;


	// Download objects
	surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    surf.downloadDescriptors(descriptors1GPU, descriptors1);
    surf.downloadDescriptors(descriptors2GPU, descriptors2);


    // drawing the results
    /*
    Mat img_matches;
    drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, good_matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);
	*/


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
	cout << "Homography completed. Time taken: " << duration_sec.count() << endl;

    // Apply homography matrix and stitch
    start = high_resolution_clock::now();
	cv::cuda::warpPerspective(img2, img_pano, H, Size(img2.cols + img1.cols, img2.rows));
	GpuMat half = img_pano(Rect(0, 0, Mat(img1).cols, Mat(img1).rows));
	img1.copyTo(half);
	end = high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	cout << "Stitching completed. Time taken: " << duration_sec.count() << endl;
	return img_pano;
}
