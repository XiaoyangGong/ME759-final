#include <string>
#include <iostream>
#include <vector>
#include <experimental/filesystem>
#include<time.h> 

#include "opencv2/core.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "opencv2/opencv_modules.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

#include <chrono>
#include <ratio>
#include <cmath>
using std::chrono::high_resolution_clock;
using std::chrono::duration;

using namespace std;
using namespace cv;
using namespace cv::cuda;
namespace fs = std::experimental::filesystem;

int main() {
	high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    const int SIZE = 640;
    const int ITE = 10;
    srand(time(0)); 
    
    std::string path = "../images/workload640_480";
    vector<string> files;
    for (const auto & entry : fs::directory_iterator(path)){
        files.push_back(entry.path());
    }

    std::vector<cv::Mat> imgs;
    for(auto file : files){
    	cout << file << endl;
    	Mat img = imread(file, IMREAD_GRAYSCALE);
    	imgs.push_back(img);
    }

    std::vector<float> time_fx;			// Time taken for feature extraction
   	std::vector<float> time_match;		// Time taken for feature matching
   	std::vector<float> time_warp;			// Time taken for perspective warping

    std::vector<float> time_gpu_fx;			// Time taken for feature extraction
   	std::vector<float> time_gpu_match;		// Time taken for feature matching
   	std::vector<float> time_gpu_warp;			// Time taken for perspective warping


    // Start testing...
    for(int i = 0; i <= 24; i++){
    	cout << "Size: " << (i+1) * SIZE << endl;
    	int rand_index = rand() % files.size();
      Mat mcat = imgs[rand_index];
      Mat firstImg = mcat;
      for(int j = 0; j < i; j++){
          rand_index = rand() % files.size();
          hconcat(mcat, imgs[rand_index], mcat);
      }
      GpuMat gpuMcat;
      gpuMcat.upload(mcat);
      //imshow("img", Mat(gpuMcat));
      //waitKey(0);

      int minHessian = 400;
      Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
      cv::cuda::SURF_CUDA surf(minHessian);

      Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      Ptr<cv::cuda::DescriptorMatcher> gpu_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());

      std::vector<KeyPoint> keypoints_first, keypoints;
      Mat descriptors_first, descriptors;
      
      GpuMat keypoints_first_GPU, keypoints_GPU;
      GpuMat descriptors_first_GPU, descriptors_GPU;
    	// Start feature extraction timing test...
    	// Repeat for 30 times and take average...
      cout << "Start CPU feature extraction test..." << endl;
      float time = 0;
      for(int k = 0; k < ITE; k++){
        start = high_resolution_clock::now();
        detector->detectAndCompute(mcat, noArray(), keypoints, descriptors);  
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        time += duration_sec.count();
            //cout << "Time taken: " << duration_sec.count() << endl;
    }
    time_fx.push_back(time / ITE);
    cout << "CPU Feature extraction test finished..." << endl;

    cout << "Start GPU feature extraction test..." << endl;
    time = 0;
    for(int k = 0; k < ITE; k++){
        start = high_resolution_clock::now();
        surf(gpuMcat, GpuMat(), keypoints_GPU, descriptors_GPU);  
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        time += duration_sec.count();
            //cout << "Time taken: " << duration_sec.count() << endl;
    }
    time_gpu_fx.push_back(time / ITE);
    cout << "GPU Feature extraction test finished..." << endl;


    // Start feature matching timing test...
    // Match image with first sub image
    cout << "Start CPU feature matching test..." << endl;
    time = 0;
    detector->detectAndCompute(firstImg, noArray(), keypoints_first, descriptors_first);
    std::vector<std::vector<DMatch>> knn_matches;
    //std::vector<DMatch> good_matches;
    for(int k = 0; k < ITE; k++){
        start = high_resolution_clock::now();
        matcher->knnMatch(descriptors_first, descriptors, knn_matches, 2);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        time += duration_sec.count();
        //cout << "Time taken for image matching: " << duration_sec.count() << endl;
    }
    time_match.push_back(time / ITE);
    cout << "CPU feature matching test finished..." << endl;


    cout << "Start GPU feature matching test..." << endl;
    time = 0;
    surf(gpuMcat, GpuMat(), keypoints_first_GPU, descriptors_first_GPU);  
    std::vector<std::vector<DMatch>> knn_matches_GPU;
    //std::vector<DMatch> good_matches;
    for(int k = 0; k < ITE; k++){
        start = high_resolution_clock::now();
        gpu_matcher->knnMatch(descriptors_first_GPU, descriptors_GPU, knn_matches_GPU, 2);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        time += duration_sec.count();
        //cout << "Time taken for image matching: " << duration_sec.count() << endl;
    }
    time_gpu_match.push_back(time / ITE);
    cout << "GPU feature matching test finished..." << endl;
    /*
    // Ratio test
    const float ratio_thresh = 0.75f;
    for (int k = 0; k < knn_matches.size(); k++){
        if (knn_matches[k][0].distance < ratio_thresh * knn_matches[k][1].distance){
            good_matches.push_back(knn_matches[k][0]);
                // TODO create match_score var from distance
        }
    }
    */
        /*
        // drawing the results
        Mat img_matches;
        drawMatches(firstImg, keypoints_first, mcat, keypoints, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imshow("Matches", img_matches);
        waitKey(0);
        */

    // Skip real homography computation because our input images has no perspective difference
    // Creat 3 x 3 pseudo-homography matrix as input for perspective warping
    // H =  [1, 0, randX];
    //      [0, 1, randY];
    //      [0, 0, 1];

    // Start CPU perspective warping timing test
    cout << "Start CPU perspective warping test..." << endl;
    time = 0;
    Mat img_pano;
    for(int k = 0; k < ITE; k++){
        float homography_data[9] = {1, 0, (float)(rand() % mcat.cols), 0, 1, (float)(rand() % mcat.rows), 0, 0, 1};
        Mat H = Mat(3, 3, CV_32F, homography_data);
        start = high_resolution_clock::now();
        cv::warpPerspective(mcat, img_pano, H, Size(mcat.cols * 2, mcat.rows));
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Time taken for perspective warping is: " << duration_sec.count() << endl;
        time += duration_sec.count();
    }
    time_warp.push_back(time / ITE);
    cout << "CPU perspective warping test finished..." << endl;

    cout << "Start GPU perspective warping test..." << endl;
    time = 0;
    GpuMat img_pano_gpu;
    for(int k = 0; k < ITE; k++){
        float homography_data[9] = {1, 0, (float)(rand() % mcat.cols), 0, 1, (float)(rand() % mcat.rows), 0, 0, 1};
        Mat H = Mat(3, 3, CV_32F, homography_data);
        start = high_resolution_clock::now();
        cv::cuda::warpPerspective(gpuMcat, img_pano_gpu, H, Size(mcat.cols * 2, mcat.rows));
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Time taken for perspective warping is: " << duration_sec.count() << endl;
        time += duration_sec.count();
    }
    time_gpu_warp.push_back(time / ITE);
    cout << "GPU perspective warping test finished..." << endl;
}

for(auto item : time_fx){
    cout << "CPU feature extraction: " << item << endl;
}
for(auto item : time_gpu_fx){
    cout << "GPU feature extraction: " << item << endl;
}

for(auto item : time_match){
    cout << "CPU feature matching: " << item << endl;
}
for(auto item : time_gpu_match){
    cout << "GPU feature matching: " << item << endl;
}


for(auto item : time_warp){
    cout << "CPU perspective warping: " << item << endl;
}
for(auto item : time_gpu_warp){
    cout << "GPU perspective warping: " << item << endl;
}



}



/*
1
2
4
8
16
32
64
128
256
512
1024

*/
