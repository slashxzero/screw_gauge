#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
RNG rng(12345);
//=======================================================
struct stLINE_INFO
{
    unsigned int line_no;
    unsigned int b2w_num;
    unsigned int w2b_num;
    unsigned int  max_black_length;
};
//=======================================================
Mat src, src_r90, small_src, small_src_r90;
vector<stLINE_INFO> v_line_info, h_line_info;
vector< vector<Point> >b2w_points;
vector< vector<Point> >w2b_points;

//=======================================================

Mat src_gray;
Mat dst, detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;

Point center;
double rotate_angle;
int thresh = 200;
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";

// Function declarations
void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);
//=======================================================
//=======================================================
void Determine_Line_Info(Mat gray, bool vertical)
{
    vector<Point> w2b;
    vector<Point> b2w;
    int max_b2w_num = 0;
    int max_w2b_num = 0;
    int b2w_interval = 0;
    int w2b_interval = 0;
    int max_black_length;
    Mat dst;
    gray.copyTo(dst);

    Point temp;
    if (vertical) v_line_info.clear();
    else h_line_info.clear();
    b2w_points.clear();
    w2b_points.clear();

    for(int i = 0 ; i < gray.rows; i++)
    {
        w2b.clear();
        b2w.clear();
        stLINE_INFO line_info;
        int b2w_num = 0;
        int w2b_num = 0;
        max_black_length = 0;
        //cout << "line[" << i << "]:";
        for(int j = 0; j < gray.cols-1; j++)
        {

            int p0 = gray.at<uchar>(i,j);
            int p1 = gray.at<uchar>(i,j+1);
            int diff_gray = p1 - p0;
            int black_length;

            if (diff_gray >= 128)
            {
                temp.x = j;
                temp.y = i;
                b2w.push_back(temp);
                b2w_num++;
                black_length = b2w[b2w_num-1].x - w2b[b2w_num-1].x;
                if (black_length > max_black_length) max_black_length = black_length;
            }
            else if (diff_gray <= -128)
            {
                temp.x = j;
                temp.y = i;
                w2b.push_back(temp);
                w2b_num++;
            }

        }
        b2w_points.push_back(b2w);
        w2b_points.push_back(w2b);
        if (max_black_length > 0)
        {
            cout << "b2w_num:" << b2w_num << endl;
            cout << "w2b_num:" << w2b_num << endl;
            cout << "Line " << i << ", " << "max_black_length:" << max_black_length << endl;
            line_info.line_no = i;
            line_info.b2w_num = b2w_num;
            line_info.w2b_num = w2b_num;
            line_info.max_black_length = max_black_length;

            if (vertical) v_line_info.push_back(line_info);
            else h_line_info.push_back(line_info);
        }
    }
}
//=======================================================
void Morph(Mat org, Mat& dst)
{
    ///Element: 0:Rect  1:Cross  2:Ellipse
    int morph_elem = 2;
    int morph_size = 1;
    ///0: Opening  1: Closing  2: Gradient  3: Top Hat  4: Black Hat
    int morph_operator = 0;
    int operation = morph_operator + 2;
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx(org, dst, operation, element);
}
//=======================================================
int main(int argc, char** argv)
{
    // Load image
    CommandLineParser parser(argc, argv, "{@input | ../data/pca_test1.jpg | input image}");
    parser.printMessage();
    Mat src = imread(parser.get<String>("@input"));
    Mat gray;
    Mat bw;
    // Check if image is loaded successfully
    if(src.empty())
    {
        cout << "Problem loading image!!!" << endl;
        return EXIT_FAILURE;
    }

    cvtColor(src, src, COLOR_BGR2GRAY);
    rotate(src, src_r90, ROTATE_90_CLOCKWISE);
    resize(src, small_src, Size(), 0.25, 0.25, INTER_LINEAR);
    resize(src_r90, small_src_r90, Size(), 0.25, 0.25, INTER_LINEAR);

    imshow("src", src);
    imshow("src_r90", src_r90);
    imshow("small_src", small_src);
    imshow("small_src_r90", small_src_r90);

    Mat mor;
    Morph(small_src, mor);
    threshold(mor, mor, 127, 255, THRESH_BINARY | THRESH_OTSU);
    Determine_Line_Info(mor, 1);
    //imshow("mor", mor);
    Morph(small_src_r90, mor);
    threshold(mor, mor, 127, 255, THRESH_BINARY | THRESH_OTSU);
    Determine_Line_Info(mor, 0);

    // Convert image to binary
    threshold(mor, bw, 127, 255, THRESH_BINARY | THRESH_OTSU);

    blur(bw, detected_edges, Size(1, 1) );
    Canny(detected_edges, dst, lowThreshold, lowThreshold*ratio, kernel_size );
    imshow("Canny edge", dst );

    // Find all the contours in the thresholded image
    /*
    vector<vector<Point> > contours;
    findContours(bw, contours, RETR_LIST, CHAIN_APPROX_NONE);
    cout << "There are " << contours.size() << "contours." << endl;
    for (size_t i = 0; i < contours.size(); i++)
    {

        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        cout << "area:" << area << endl;
        // Ignore contours that are too small or too large
        if (area < 1e2)
            continue;
        vector<RotatedRect> minRect(contours.size() );
        minRect[i] = minAreaRect( contours[i] );

        // Draw each contour only for visualisation purposes
        drawContours(bw, contours, static_cast<int>(i), Scalar(0, 0, 255), 2);
        // rotated rectangle
        Point2f rect_points[4];
        minRect[i].points( rect_points );
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        for ( int j = 0; j < 4; j++ )
        {
            line(bw, rect_points[j], rect_points[(j+1)%4], color );
        }
        break;
    }
    imshow("output", bw);
    */

    int top, bottom, left, right;
    int borderType = BORDER_CONSTANT;
    // Initialize arguments for the filter
    top = (int) (0.1*bw.rows);
    bottom = top;
    left = (int) (0.1*bw.cols);
    right = left;

    Scalar value( 255, 255, 255 );
    copyMakeBorder( bw, dst, top, bottom, left, right, borderType, value );
    imshow("Border", dst );

    waitKey();
    return 0;
}
//=======================================================
