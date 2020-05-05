#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <chrono>
#include <ratio>
#include <cmath>
using std::chrono::high_resolution_clock;
using std::chrono::duration;


using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "\nThis program demonstrates using features detector, descriptor extractor and Matcher" << endl;
    cout << "\nUsage:\n\tpanaroma_stitcher <number of images>" << endl;
}


int main(int argc, char* argv[])
{
    srand(time(0));
    if (argc > 2)
    {
        help();
        return -1;
    }

    int n = 0;
    try{
        n = stoi(argv[1]);
        if(n < 1)
            exit(1);
    }
    catch (std::invalid_argument const &e){
        help();
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    const int ITE = 10; // Repeat test for 10 times
    // Create n workload from workload1000 directory, 7 pairs of pano images
    GpuMat* imgs = new GpuMat[n];
    // Construct the first image
    Mat img1;
    string img_path;
    int img_num = random() % 14;
    img_num = (img_num == 0) ? 1 : img_num;
    img_path = "../images/workload1000/" + to_string(img_num) + ".png";
    img1 = imread(img_path, IMREAD_GRAYSCALE);
    CV_Assert(!img1.empty());

    Mat img2;
    img_num = random() % 14;
    img_num = (img_num == 0) ? 1 : img_num;
    img_num = (img_num % 2 == 0) ? img_num - 1 : img_num;
    img_path = "../images/workload1000/" + to_string(img_num) + ".png";
    img2 = imread(img_path, IMREAD_GRAYSCALE);
    CV_Assert(!img2.empty());
    Mat inputImg;
    hconcat(img1, img2, inputImg);
    CV_Assert(!inputImg.empty());

    imgs[0].upload(inputImg);

    // Construct the rest of input images
    for(int i = 1; i < n; i++){
        Mat left;
        img_path = "../images/workload1000/" + to_string(img_num+1) + ".png";
        left = imread(img_path, IMREAD_GRAYSCALE);        
        CV_Assert(!left.empty());

        Mat right;
        img_num = random() % 14;
        img_num = (img_num == 0) ? 1 : img_num;
        img_num = (img_num % 2 == 0) ? img_num - 1 : img_num;
        img_path = "../images/workload1000/" + to_string(img_num) + ".png";
        right = imread(img_path, IMREAD_GRAYSCALE);
        CV_Assert(!right.empty());
        hconcat(left, right, inputImg);

        imgs[i].upload(inputImg);
    }



    float time = 0;
    for(int k = 0; k < ITE; k++){
    // detecting keypoints & computing descriptors for all imgs
        start = high_resolution_clock::now();
        double minHessian = 400;
        SURF_CUDA surf(minHessian);

    //vector<GpuMat>* keypointsGPU = new vector<GpuMat>[n];
        GpuMat* keypointsGPU = new GpuMat[n];
        GpuMat* descriptorsGPU = new GpuMat[n];
        vector<KeyPoint>* keypoints = new vector<KeyPoint>[n];
        vector<float>* descriptors = new vector<float>[n];

        for(int i = 0; i < n; i++){
            surf(imgs[i], GpuMat(), keypointsGPU[i], descriptorsGPU[i]);
        }



    // Match descriptors with FLANN based matcher
    // Match img1 & img2; img2 & img3; ... ; img(n-1) & img(n)
        Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
        vector<vector<cv::DMatch>>* knn_matches = new vector<vector<cv::DMatch>>[n-1];
        for(int i = 0; i < n-1; i++){
            matcher->knnMatch(descriptorsGPU[i], descriptorsGPU[i+1], knn_matches[i], 2);
        }

    // Filter matches using the Lowe's ratio test
    // For n images, number of matches is n - 1
    // Can use OpenMP
        const float ratio_thresh = 0.7f;
        vector<DMatch>* good_matches = new vector<DMatch>[n-1];
        for(int i = 0; i < n-1; i++){
            for (size_t j = 0; j < knn_matches[i].size(); j++){
                if (knn_matches[i][j][0].distance < ratio_thresh * knn_matches[i][j][1].distance){
                    good_matches[i].push_back(knn_matches[i][j][0]);
                }
            }
        }

    // Download objects
        for(int i = 0; i < n; i++){
            surf.downloadKeypoints(keypointsGPU[i], keypoints[i]);
            surf.downloadDescriptors(descriptorsGPU[i], descriptors[i]);
        }

    // Localize the object and get keypoints from the good matches
    // Each non-head nor non-tail img has two set of matching keypoints. 
    // E.g. img1 has matching keypoints with img0, and matching keypoints with img2
    // Let each img take 2 spot in array objs. E.g. img0 taks objs[0], objs[1]
        vector<Point2f>* objs = new vector<Point2f>[2*n-2];
        for(int i = 0; i < n-1; i++){
            for(int j = 0; j < good_matches[i].size(); j++){
            //-- Get the keypoints from the good matches
                objs[2*i].push_back(keypoints[i][ good_matches[i][j].queryIdx ].pt );
                objs[2*i+1].push_back(keypoints[i+1][ good_matches[i][j].trainIdx ].pt );
            }
        }

    // Find homography matrix
    // Note: order of obj2, obj1 does matter
    // Use obj1/img (the leftmost image) as reference perspective
        Mat* Hs = new Mat[n-1];
        for(int i = 0; i < n-1; i++){
            Hs[i] = cv::findHomography(objs[2*i+1], objs[2*i], RANSAC);
        }

        GpuMat img_pano = imgs[n-1];
    // Start from right-most imgs. Shifting perpespetive while proceeding
        for(int i = n-1; i > 0; i--){
            cv::cuda::warpPerspective(img_pano, img_pano, Hs[i-1], Size(img_pano.cols + imgs[i-1].cols, img_pano.rows));
            GpuMat left_half = img_pano(Rect(0, 0, imgs[i-1].cols, imgs[i-1].rows));  
            imgs[i-1].copyTo(left_half);
        }
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        time += duration_sec.count();
    }
    float avg_time = time / ITE;
    cout << "Time taken: " << to_string(avg_time) << endl;
    return 0;
}


#else

int main()
{
    std::cerr << "Error: Panaroma stitching requires OpenCV xfeatures2d module" << std::endl;
    return 0;
}

#endif
