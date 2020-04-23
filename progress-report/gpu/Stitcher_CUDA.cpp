#include "Stitcher_CUDA.hpp"
using namespace std;
using namespace cv;
using namespace cv::cuda;
	// constructor
Stitcher_CUDA :: Stitcher(){
	minHessian = 400;
	surf(minHessian);
	matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
}

GpuMat Stitcher_CUDA :: stitch(GpuMat& img1, GpuMat& img2){
	// detecting keypoints & computing descriptors
    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    	// Match descriptors with FLANN based matcher
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    matcher->match(descriptors1GPU, descriptors2GPU, matches);
    

    // TODO: find CUDA KNN matcher
    /*
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
	*/

	// Download objects
	surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    surf.downloadDescriptors(descriptors1GPU, descriptors1);
    surf.downloadDescriptors(descriptors2GPU, descriptors2);

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

    // Apply homography matrix and stitch
	warpPerspective(img2, img_pano, H, Size(img2.cols + img1.cols, img2.rows));
	Mat half = img_pano(Rect(0, 0, img1.cols, img1.rows));
	img1.copyTo(half);
		//imshow("Panorama", img_pano);
		//waitKey(0);
	return img_pano;
}
