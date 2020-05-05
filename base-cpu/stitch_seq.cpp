#include "Stitcher.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
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
    cout << "\nUsage:\n\tpanaroma_stitcher <number of images> <image1> <image2> ... <imageN>" << endl;
}


int main(int argc, char* argv[]){
    cout << "\n Start Stitching: Sequential mode" << endl;

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

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    Mat* imgs = new Mat[n];

    // TODO check input image size matches
    for (int i = 0; i < n; ++i){
        imgs[i] = imread(argv[i+2], IMREAD_GRAYSCALE);
        CV_Assert(!imgs[i].empty());
    }

    Stitcher* st = new Stitcher();
    Mat img_intermed = imgs[n-1];


    float time = 0;
    for(int j = 0; j < 10; j++){



    start = high_resolution_clock::now();
    for(int i = n-1; i > 0; i--){
    	img_intermed = st->stitch(imgs[i-1], img_intermed);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    time += duration_sec.count();

    }
    cout << "Panorama stitching completed. Time taken: " << time / 10 << endl;
    //Mat img_pano = img_intermed;
    //imshow("Pano", img_pano);
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
