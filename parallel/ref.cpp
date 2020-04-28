#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>


void example_with_full_gpu(const cv::Mat &img1, const cv::Mat img2) {
//Upload from host memory to gpu device memeory
cv::cuda::GpuMat img1_gpu(img1), img2_gpu(img2);
cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

//Convert RGB to grayscale as gpu detectAndCompute only allow grayscale GpuMat
cv::cuda::cvtColor(img1_gpu, img1_gray_gpu, 6);
cv::cuda::cvtColor(img2_gpu, img2_gray_gpu, 6);

//Create a GPU ORB feature object
//blurForDescriptor=true seems to give better results
//http://answers.opencv.org/question/10835/orb_gpu-not-as-good-as-orbcpu/
cv::cuda::SURF_CUDA surf;
//cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();

cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
//Detect ORB keypoints and extract descriptors on train image (box.png)
surf(img1_gray_gpu, cv::cuda::GpuMat(), keypoints1_gpu, descriptors1_gpu);
// orb->detectAndComputeAsync(img1_gray_gpu, cv::cuda::GpuMat(), keypoints1_gpu, descriptors1_gpu);
std::vector<cv::KeyPoint> keypoints1;
//Convert from CUDA object to std::vector<cv::KeyPoint>
surf.downloadKeypoints(keypoints1_gpu, keypoints1);
//orb->convert(keypoints1_gpu, keypoints1);
std::cout << "keypoints1=" << keypoints1.size() << " ; descriptors1_gpu=" << descriptors1_gpu.rows 
    << "x" << descriptors1_gpu.cols << std::endl;

cv::cuda::GpuMat keypoints2_gpu, descriptors2_gpu;
//Detect ORB keypoints and extract descriptors on query image (box_in_scene.png)
//The conversion from internal data to std::vector<cv::KeyPoint> is done implicitly in detectAndCompute()
surf(img2_gray_gpu, cv::cuda::GpuMat(), keypoints2_gpu, descriptors2_gpu);
std::vector<cv::KeyPoint> keypoints2;
surf.downloadKeypoints(keypoints2_gpu, keypoints2);
std::cout << "keypoints2=" << keypoints2.size() << " ; descriptors2_gpu=" << descriptors2_gpu.rows 
    << "x" << descriptors2_gpu.cols << std::endl;

//Create a GPU brute-force matcher with Hamming distance as we use a binary descriptor (ORB)
cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());

std::vector<std::vector<cv::DMatch> > knn_matches;
//Match each query descriptor to a train descriptor
matcher->knnMatch(descriptors2_gpu, descriptors1_gpu, knn_matches, 2);
std::cout << "knn_matches=" << knn_matches.size() << std::endl;

std::vector<cv::DMatch> matches;
//Filter the matches using the ratio test
for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
    if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.8) {
        matches.push_back((*it)[0]);
    }
}

cv::Mat imgRes;
//Display and save the image with matches
cv::drawMatches(img2, keypoints2, img1, keypoints1, matches, imgRes);
cv::imshow("imgRes", imgRes);
cv::imwrite("GPU_ORB-matching.png", imgRes);

cv::waitKey(0); 
}

void example_with_gpu_matching(const cv::Mat &img1, const cv::Mat img2) {
//Create a CPU ORB feature object
cv::Ptr<cv::Feature2D> orb = cv::ORB::create();

std::vector<cv::KeyPoint> keypoints1;
cv::Mat descriptors1;
//Detect ORB keypoints and extract descriptors on train image (box.png)
orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
std::cout << "keypoints1=" << keypoints1.size() << " ; descriptors1=" << descriptors1.rows 
    << "x" << descriptors1.cols << std::endl;

std::vector<cv::KeyPoint> keypoints2;
cv::Mat descriptors2;
//Detect ORB keypoints and extract descriptors on query image (box_in_scene.png)
orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
std::cout << "keypoints2=" << keypoints2.size() << " ; descriptors2=" << descriptors2.rows 
    << "x" << descriptors2.cols << std::endl;

//Create a GPU brute-force matcher with Hamming distance as we use a binary descriptor (ORB)
cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

//Upload from host memory to gpu device memeory
cv::cuda::GpuMat descriptors1_gpu(descriptors1), descriptors2_gpu;
//Upload from host memory to gpu device memeory (another way to do it)
descriptors2_gpu.upload(descriptors2);

std::vector<std::vector<cv::DMatch> > knn_matches;
//Match each query descriptor to a train descriptor
matcher->knnMatch(descriptors2_gpu, descriptors1_gpu, knn_matches, 2);
std::cout << "knn_matches=" << knn_matches.size() << std::endl;

std::vector<cv::DMatch> matches;
//Filter the matches using the ratio test
for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
    if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.8) {
        matches.push_back((*it)[0]);
    }
}

cv::Mat imgRes;
//Display and save the image with matches
cv::drawMatches(img2, keypoints2, img1, keypoints1, matches, imgRes);
cv::imshow("imgRes", imgRes);   
cv::imwrite("CPU_ORB+GPU_matching.png", imgRes);

cv::waitKey(0); 
}

int main() {
std::cout << "OpenCV version=" << std::hex << CV_VERSION << std::dec << std::endl;

cv::Mat img1, img2;
img1 = cv::imread("./matching/altera_in_scene.jpg");
img2 = cv::imread("./matching/altera.jpg");

example_with_full_gpu(img1, img2);
example_with_gpu_matching(img1, img2);

return 0;
}
