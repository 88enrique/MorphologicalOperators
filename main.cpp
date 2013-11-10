/**
    Opencv example code: different morphologic operations over an image
    Enrique Marin
    88enrique@gmail.com
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(){

    // Variables
    Mat src, src_gray, src_thresh;
    Mat element = getStructuringElement(MORPH_ELLIPSE,Size(5,5), Point(3,3));

    // Load an image
    src = imread("../Images/coins1.jpg");
    if( !src.data ){
        return -1;
    }

    // Mat for different morphologic process
    Mat dst_erode = Mat(src.rows, src.cols, CV_8UC1);
    Mat dst_dilate = Mat(src.rows, src.cols, CV_8UC1);
    Mat dst_open = Mat(src.rows, src.cols, CV_8UC1);
    Mat dst_close = Mat(src.rows, src.cols, CV_8UC1);
    Mat dst_grad = Mat(src.rows, src.cols, CV_8UC1);
    Mat dst_top = Mat(src.rows, src.cols, CV_8UC1);
    Mat dst_black = Mat(src.rows, src.cols, CV_8UC1);

    // Remove noise by blurring with a Gaussian filter
    GaussianBlur(src, src, Size(3,3), 0.1, 0, BORDER_DEFAULT);

    // Convert the image to grayscale
    cvtColor(src, src_gray, CV_RGB2GRAY);

    // Otsu Threshold
    threshold(src_gray, src_thresh, 0, 255, CV_THRESH_OTSU);

    // Create window
    namedWindow("Original - Otsu", CV_WINDOW_NORMAL);
    namedWindow("Dilate", CV_WINDOW_NORMAL);
    namedWindow("Erode", CV_WINDOW_NORMAL);
    namedWindow("Open", CV_WINDOW_NORMAL);
    namedWindow("Close", CV_WINDOW_NORMAL);
    namedWindow("Fill Holes", CV_WINDOW_NORMAL);
    namedWindow("Gradient", CV_WINDOW_NORMAL);
    namedWindow("Top Hat", CV_WINDOW_NORMAL);
    namedWindow("Black Hat", CV_WINDOW_NORMAL);

    // Morphology operations
    morphologyEx(src_thresh, dst_dilate, MORPH_DILATE, element);
    morphologyEx(src_thresh, dst_erode, MORPH_ERODE, element);
    morphologyEx(src_thresh, dst_open, MORPH_OPEN, element);
    morphologyEx(src_thresh, dst_close, MORPH_CLOSE, element);
    morphologyEx(src_thresh, dst_grad, MORPH_GRADIENT, element);
    morphologyEx(src_thresh, dst_top, MORPH_TOPHAT, element);
    morphologyEx(src_thresh, dst_black, MORPH_BLACKHAT, element);

    // Fill holes
    Mat dst_holes = dst_open;
    floodFill(dst_holes, Point(0,0), cvScalar(128));
    inRange(dst_holes, 128, 128, dst_holes);

    // Show images
    imshow("Original - Otsu", src_thresh);
    imshow("Dilate", dst_dilate);
    imshow("Erode", dst_erode);
    imshow("Open", dst_open);
    imshow("Close", dst_close);
    imshow("Fill Holes", dst_holes);
    imshow("Gradient", dst_grad);
    imshow("Top Hat", dst_top);
    imshow("Black Hat", dst_black);

    waitKey(0);

    // Release memory
    src_thresh.release();
    dst_erode.release();
    dst_dilate.release();
    dst_open.release();
    dst_close.release();
    dst_grad.release();
    dst_top.release();
    dst_black.release();
    dst_holes.release();

    return 0;
}

