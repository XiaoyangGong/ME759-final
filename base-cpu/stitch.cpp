#include <iostream>
#include <vector>
#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static void help()
{
    cout << "\nThis program demonstrates using features detector, descriptor extractor and Matcher" << endl;
    cout << "\nUsage:\n\tsurf_keypoint_matcher --left <image1> --right <image2>" << endl;
}


int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        help();
        return -1;
    }

    Mat img1, img2;
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--left")
        {
            img1 = imread(argv[++i], IMREAD_GRAYSCALE);
            CV_Assert(!img1.empty());
        }
        else if (string(argv[i]) == "--right")
        {
            img2 = imread(argv[++i], IMREAD_GRAYSCALE);
            CV_Assert(!img2.empty());
        }
        else if (string(argv[i]) == "--help")
        {
            help();
            return -1;
        }
    }

    // detecting keypoints & computing descriptors
    double minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    cout << "FOUND " << keypoints1.size() << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2.size() << " keypoints on second image" << endl;

    // Match descriptors with FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    // Can use OpenMP
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // drawing the results
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //namedWindow("matches", 0);
    //imshow("matches", img_matches);
    //imwrite("matches.jpg", img_matches);
    //waitKey(0);
    
    // Localize the object
    std::vector<Point2f> obj1;
    std::vector<Point2f> obj2;
    for( size_t i = 0; i < good_matches.size(); i++ ){
        //-- Get the keypoints from the good matches
        obj1.push_back(keypoints1[ good_matches[i].queryIdx ].pt );
        obj2.push_back(keypoints2[ good_matches[i].trainIdx ].pt );
    }

    // Find homography matrix
    // Note: order of obj2, obj1 does matter
    Mat H = cv::findHomography(obj2, obj1, RANSAC);

    // Apply homography matrix and stitch
    Mat panorama;
    warpPerspective(img2, panorama, H, Size(img2.cols * 2, img2.rows));

    Mat half = panorama(Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

    imshow("pic", panorama);
    waitKey(0);






    return 0;
}


#else

int main()
{
    std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
    return 0;
}

#endif
