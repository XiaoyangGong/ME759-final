#include "Stitcher_CUDA.hpp"
#include <omp.h>
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
    for (int i = 0; i < n; ++i){
        imgs[i].upload(imread(argv[i+2], IMREAD_GRAYSCALE));
        CV_Assert(!imgs[i].empty());
    }
    
    int ites = ceil(log2(n));

    // always work from right to left because of the way stitcher is written
    for(int i = 0; i < ites; i++){
        #pragma omp parallel for
        for(int j = n-1; j > 0; j -= (1 << (i+1))){
            Stitcher_CUDA* st = new Stitcher_CUDA();
            if(j-(1<<i) >= 0){
                imgs[j] = st->stitch(imgs[j-(1<<i)], imgs[j]);
            }
            delete st;
        }
        #pragma omp barrier
    }

    GpuMat img_pano = imgs[n-1];
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
