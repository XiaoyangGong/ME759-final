#include "Stitcher_CUDA.hpp"
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

	// detecting keypoints & computing descriptors
    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);


    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    
    // Match descriptors with FLANN based matcher
    matcher->knnMatch(descriptors1GPU, descriptors2GPU, knn_matches, 2);
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
	H = cv::findHomography(obj2, obj1, RANSAC);
	Mat cpu_img_pano;
    // Apply homography matrix and stitch
	cv::cuda::warpPerspective(img2, img_pano, H, Size(img2.cols + img1.cols, img2.rows));
	GpuMat half = img_pano(Rect(0, 0, Mat(img1).cols, Mat(img1).rows));
	img1.copyTo(half);
	return img_pano;
}
