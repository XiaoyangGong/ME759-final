#include "Stitcher.hpp"
#include <cmath>
#ifdef HAVE_OPENCV_XFEATURES2D

#include<time.h> 
#include <chrono>
#include <ratio>
#include <cmath>
using std::chrono::high_resolution_clock;
using std::chrono::duration;

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

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
        cout << "Input: " << n << " images" << endl;
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
    Mat* imgs = new Mat[n];
    // Construct the first image
    Mat img1;
    string img_path;
    int img_num = random() % 14;
    img_num = (img_num == 0) ? 1 : img_num;
    img_path = "../images/workload1000/" + to_string(img_num) + ".png";
    cout << img_path << endl;
    img1 = imread(img_path, IMREAD_GRAYSCALE);
    CV_Assert(!img1.empty());

    Mat img2;
    img_num = random() % 14;
    img_num = (img_num == 0) ? 1 : img_num;
    img_num = (img_num % 2 == 0) ? img_num - 1 : img_num;
    img_path = "../images/workload1000/" + to_string(img_num) + ".png";
    cout << img_path << endl;
    img2 = imread(img_path, IMREAD_GRAYSCALE);
    CV_Assert(!img2.empty());
    Mat inputImg;
    hconcat(img1, img2, inputImg);
    CV_Assert(!inputImg.empty());

    imgs[0] = inputImg;

    // Construct the rest of input images
    for(int i = 1; i < n; i++){
        Mat left;
        img_path = "../images/workload1000/" + to_string(img_num+1) + ".png";
        cout << img_path << endl;
        left = imread(img_path, IMREAD_GRAYSCALE);        
        CV_Assert(!left.empty());

        Mat right;
        img_num = random() % 14;
        img_num = (img_num == 0) ? 1 : img_num;
        img_num = (img_num % 2 == 0) ? img_num - 1 : img_num;
        img_path = "../images/workload1000/" + to_string(img_num) + ".png";
        cout << img_path << endl;
        right = imread(img_path, IMREAD_GRAYSCALE);
        CV_Assert(!right.empty());
        hconcat(left, right, inputImg);

        imgs[i] = inputImg;
    }
    


    // Set up test
    int ites = ceil(log2(n));
    float time = 0;
    // always work from right to left because of the way stitcher is written
    for(int k = 0; k < ITE; k++){
        start = high_resolution_clock::now();
        for(int i = 0; i < ites; i++){
        #pragma omp parallel for
            for(int j = n-1; j > 0; j -= (1 << (i+1))){
                Stitcher* st = new Stitcher();
                if(j-(1<<i) >= 0){
                    imgs[j] = st->stitch(imgs[j-(1<<i)], imgs[j]);
                }
                delete st;
            }
        #pragma omp barrier
        }
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        time += duration_sec.count();
    }

    float avg_time = time / ITE;
    cout << "Time taken: " << to_string(avg_time) << endl;
    //GpuMat img_pano = imgs[n-1];
    //imshow("Pano", Mat(img_pano));
    //waitKey(0);
    return 0;
}


#else

int main()
{
    std::cerr << "Error: Panaroma stitching requires OpenCV xfeatures2d module" << std::endl;
    return 0;
}

#endif
