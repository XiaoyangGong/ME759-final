#include "Stitcher_CUDA.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static void help()
{
    cout << "\nThis program demonstrates using features detector, descriptor extractor and Matcher" << endl;
    cout << "\nUsage:\n\tpanaroma_stitcher <number of images> <image1> <image2> ... <imageN>" << endl;
}


int main(int argc, char* argv[])
{
    if (argc < 3){
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
    for (int i = 0; i < n; ++i){
        imgs[i].upload(imread(argv[i+2], IMREAD_GRAYSCALE));
        CV_Assert(!imgs[i].empty());
    }
    
    Stitcher_CUDA* st = new Stitcher_CUDA();
    
    GpuMat img_intermed = imgs[n-1];
    for(int i = n-1; i > 0; i--){
    	img_intermed = st->stitch(imgs[i-1], img_intermed);
    }
    GpuMat img_pano = img_intermed;
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
