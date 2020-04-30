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




using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "\nThis program demonstrates using features detector, descriptor extractor and Matcher" << endl;
    cout << "\nUsage:\n\tpanaroma_stitcher <number of images> <image1> <image2> ... <imageN>" << endl;
}


int main(int argc, char* argv[])
{
    cout << "\n Start Stitching: Sequential Optimatized mode" << endl;

    if (argc < 3)
    {
        help();
        return -1;
    }

    int n = 0;
    try{
        n = stoi(argv[1]);
        cout << "Input: " << n << " images" << endl;

        if(n < 2 || argc < 2 + n){
            help();
            return -1;
        }
    }
    catch (std::invalid_argument const &e){
        help();
	}

    GpuMat* imgs = new GpuMat[n];

    // TODO check input image size matches
    for (int i = 0; i < n; ++i)
    {
        imgs[i].upload(imread(argv[i+2], IMREAD_GRAYSCALE));
        CV_Assert(!imgs[i].empty());
    }
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());


    // detecting keypoints & computing descriptors for all imgs
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
    for(int i = 0; i < n; i++){
        cout << "FOUND " << keypointsGPU[i].cols << " keypoints on image " << i << endl;
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

    // drawing the results
    Mat* img_matches = new Mat[n-1];
    for(int i = 0; i < n-1; i++){
        drawMatches(Mat(imgs[i]), keypoints[i], Mat(imgs[i+1]), keypoints[i+1], good_matches[i], img_matches[i], Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    }

    // print out matches
    for(int i = 0; i < n-1; i++){
        namedWindow(to_string(i), WINDOW_AUTOSIZE);
        imshow(to_string(i), img_matches[i]);
        waitKey(0);
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
        cout << "Find homography pair " << i << " and " << i+1 << endl;
    }

    GpuMat img_pano = imgs[n-1];
    // Start from right-most imgs. Shifting perpespetive while proceeding
    for(int i = n-1; i > 0; i--){
        cv::cuda::warpPerspective(img_pano, img_pano, Hs[i-1], Size(img_pano.cols + imgs[i-1].cols, img_pano.rows));
        GpuMat left_half = img_pano(Rect(0, 0, imgs[i-1].cols, imgs[i-1].rows));  
        imgs[i-1].copyTo(left_half);
        cout << "finish pair " << i << " and " << i+1 << endl;
    }
    imshow("Pano", Mat(img_pano));
    waitKey(0);
    return 0;
}


#else

int main()
{
    std::cerr << "Error: Panaroma stitching requires OpenCV xfeatures2d module" << std::endl;
    return 0;
}

#endif
