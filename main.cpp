#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
RNG rng(12345);

const double kDistanceCoef = 100.0;
const int kMaxMatchingSize = 50;
Mat homo_matrix;
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
void Find_Screw_Thread(Mat gray)
{
    const int MIN_THREAD_NUM = 4;
    const int THREAD_HEIGHT_RANGE = 10;
    const int MIN_PITCH = 5;

    vector<vector<Point> > w2b_all_lines;
    vector<vector<Point> > b2w_all_lines;
    vector<Point> w2b;
    vector<Point> b2w;
    int start_row = gray.rows/2;
    int start_col = gray.cols/2;
    Point LT,
          RT,
          LB,
          RB;

    Mat dst;
    gray.copyTo(dst);
    cvtColor(dst, dst, COLOR_GRAY2BGR);

    b2w_points.clear();
    w2b_points.clear();

    bool first_thread_found = false;
    for(int i = start_row ; i > 0; i--)
    {
        w2b.clear();
        b2w.clear();

        int b2w_num = 0;
        int w2b_num = 0;

        ///Scan top-half image
        Point temp;
        for(int j = 0; j < gray.cols-1; j++)
        {
            int p0 = gray.at<uchar>(i,j);
            int p1 = gray.at<uchar>(i,j+1);
            int diff_gray = p1 - p0;

            if (diff_gray >= 128)
            {
                temp.x = j;
                temp.y = i;
                b2w.push_back(temp);
                b2w_num++;

                if (!first_thread_found && b2w_num >= MIN_THREAD_NUM)
                {
                    first_thread_found = true;
                    LT.x = temp.x;
                    LT.y = temp.y;
                    RT.x = b2w[0].x;
                    RT.y = b2w[0].y;
                }
            }
            else if (diff_gray <= -128)
            {
                temp.x = j;
                temp.y = i;
                w2b.push_back(temp);
                w2b_num++;
            }
        }
        if (b2w_num >= MIN_THREAD_NUM)
        {
            bool point_swap = false;
            if (abs(b2w[0].y - LT.y) <= THREAD_HEIGHT_RANGE &&
                (LT.x - b2w[0].x) > MIN_PITCH)
            {
                LT.x = b2w[0].x;
                LT.y = b2w[0].y;
                point_swap = true;
            }

            if (abs(b2w[b2w_num-2].y - RT.y) <= THREAD_HEIGHT_RANGE &&
                (b2w[b2w_num-2].x - RT.x) > MIN_PITCH)
            {
                RT.x = b2w[b2w_num-2].x;
                RT.y = b2w[b2w_num-2].y;
                point_swap = true;
            }

            if (point_swap)
            {
                //Mat draw_dst(dst);
                //circle(draw_dst, LT, 3, Scalar(255, 0, 0), 1);
                //circle(draw_dst, RT, 3, Scalar(0, 255, 0), 1);
                //imshow("thread conner", draw_dst);
                //waitKey();
            }
        }
    }

    first_thread_found = false;
    for(int i = start_row+1; i < gray.rows; i++)
    {
        w2b.clear();
        b2w.clear();

        int b2w_num = 0;
        int w2b_num = 0;

        ///Scan bottom-half image
        Point temp;
        for(int j = 0; j < gray.cols-1; j++)
        {
            int p0 = gray.at<uchar>(i,j);
            int p1 = gray.at<uchar>(i,j+1);
            int diff_gray = p1 - p0;

            if (diff_gray >= 128)
            {
                temp.x = j;
                temp.y = i;
                b2w.push_back(temp);
                b2w_num++;

                if (!first_thread_found && b2w_num >= MIN_THREAD_NUM)
                {
                    first_thread_found = true;
                    LB.x = temp.x;
                    LB.y = temp.y;
                    RB.x = b2w[0].x;
                    RB.y = b2w[0].y;
                }
            }
            else if (diff_gray <= -128)
            {
                temp.x = j;
                temp.y = i;
                w2b.push_back(temp);
                w2b_num++;
            }
        }
        if (b2w_num >= MIN_THREAD_NUM)
        {
            bool point_swap = false;
            if (abs(b2w[0].y - LB.y) <= THREAD_HEIGHT_RANGE &&
                (LB.x - b2w[0].x) > MIN_PITCH)
            {
                LB.x = b2w[0].x;
                LB.y = b2w[0].y;
                point_swap = true;
            }

            if (abs(b2w[b2w_num-2].y - RB.y) <= THREAD_HEIGHT_RANGE &&
                (b2w[b2w_num-2].x - RB.x) > MIN_PITCH)
            {
                RB.x = b2w[b2w_num-2].x;
                RB.y = b2w[b2w_num-2].y;
                point_swap = true;
            }

            if (point_swap)
            {
                //Mat draw_dst(dst);
                //circle(draw_dst, LB, 3, Scalar(128, 128, 0), 1);
                //circle(draw_dst, RB, 3, Scalar(0, 128, 128), 1);
                //imshow("thread conner", draw_dst);
                //waitKey();
            }
        }
    }

    Mat draw_dst(dst);
    circle(draw_dst, LT, 3, Scalar(255, 0, 0), 2);
    circle(draw_dst, RT, 3, Scalar(0, 255, 0), 2);
    circle(draw_dst, LB, 3, Scalar(128, 128, 0), 2);
    circle(draw_dst, RB, 3, Scalar(0, 128, 128), 2);
    imshow("thread conner", draw_dst);
}
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
    if (vertical)
        v_line_info.clear();
    else
        h_line_info.clear();
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
                if (black_length > max_black_length)
                    max_black_length = black_length;
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

            if (vertical)
                v_line_info.push_back(line_info);
            else
                h_line_info.push_back(line_info);
        }
    }
}
//=======================================================
void Morph(Mat org, Mat& dst)
{
    ///Element: 0:Rect  1:Cross  2:Ellipse
    int morph_elem = 2;
    int morph_size = org.cols/266;
    ///0: Opening  1: Closing  2: Gradient  3: Top Hat  4: Black Hat
    int morph_operator = 0;
    int operation = morph_operator + 2;
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx(org, dst, operation, element);
}

//=======================================================
inline void detect_and_compute(string type, Mat& img, vector<KeyPoint>& kpts, Mat& desc)
{
    if (type.find("fast") == 0)
    {
        cout << "fast" << endl;
        type = type.substr(4);
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(img, kpts);
    }
    if (type.find("blob") == 0)
    {
        cout << "blob" << endl;
        type = type.substr(4);
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
        detector->detect(img, kpts);
    }
    if (type == "orb")
    {
        cout << "orb" << endl;
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "brisk")
    {
        cout << "brisk" << endl;
        Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "kaze")
    {
        cout << "kaze" << endl;
        Ptr<KAZE> kaze = KAZE::create();
        kaze->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "akaze")
    {
        cout << "akaze" << endl;
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(img, Mat(), kpts, desc);
    }

}
//=======================================================
inline void match(string type, Mat& desc1, Mat& desc2, vector<DMatch>& matches)
{
    matches.clear();
    if (type == "bf")
    {
        BFMatcher desc_matcher(NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn")
    {
        BFMatcher desc_matcher(NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i)
        {
            if (!vmatches[i].size())
            {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance)
    {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize)
    {
        matches.pop_back();
    }
}
//=======================================================
inline void findKeyPointsHomography(vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2,
                                    vector<DMatch>& matches, vector<char>& match_mask)
{
    if (static_cast<int>(match_mask.size()) < 3)
    {
        return;
    }
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    homo_matrix = findHomography(pts1, pts2, RANSAC, 4, match_mask);
}
//=======================================================
void Image_Match(Mat image1, Mat image2)
{
    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2;

    Mat desc1;
    Mat desc2;

    vector<DMatch> matches;

    string desc_type("akaze");
    string match_type("knn");


    detect_and_compute(desc_type, image1, kpts1, desc1);
    detect_and_compute(desc_type, image2, kpts2, desc2);
    match(match_type, desc1, desc2, matches);

    vector<char> match_mask(matches.size(), 1);
    findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

    Mat res;
    drawMatches(image1, kpts1, image2, kpts2, matches, res, Scalar::all(-1),
                    Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("result", res);

    Mat image1_warp;
    warpPerspective(image1, image1_warp, homo_matrix, image2.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
    //imshow("img1_warp", img1_warp);

    Mat dst;
    subtract(image2, image1_warp, dst, noArray(), 1);
    //imshow("subtract", dst);

    Mat bin_H;
    bin_H = (dst < -100) | (dst > 100);
    imshow("bin_H", bin_H);
}
//=======================================================
void Erosion(Mat erosion_src, Mat& erosion_dst)
{
    int erosion_elem = 2;
    int erosion_size = 2;
    int erosion_type = 0;

    if(erosion_elem == 0)
    {
        erosion_type = MORPH_RECT;
    }
    else if( erosion_elem == 1 )
    {
        erosion_type = MORPH_CROSS;
    }
    else if( erosion_elem == 2)
    {
        erosion_type = MORPH_ELLIPSE;
    }
    Mat element = getStructuringElement( erosion_type,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );
    erode(erosion_src, erosion_dst, element);
    imshow( "Erosion Demo", erosion_dst);
}
//=======================================================

int main(int argc, char** argv)
{
    // Load image
    //CommandLineParser parser(argc, argv, "{@input | ../data/pca_test1.jpg | input image}");
    //parser.printMessage();
    //Mat src = imread(parser.get<String>("@input"));


    string img_file1;
    string img_file2;

    if (argc >= 2) img_file1 = argv[1];
    if (argc >= 3) img_file2 = argv[2];
    cout << img_file1 << endl;
    Mat src = imread(img_file1);

    if (argc >= 3)
    {
        Mat image_to_match = imread(img_file2);
        Image_Match(src, image_to_match);
    }

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
    //imshow("src_r90", src_r90);
    //imshow("small_src", small_src);
    //imshow("small_src_r90", small_src_r90);

    Mat mor;
    Morph(small_src, mor);
    threshold(mor, mor, 127, 255, THRESH_BINARY | THRESH_OTSU);
    //Determine_Line_Info(mor, 1);
    //imshow("mor", mor);
    Morph(small_src_r90, mor);
    threshold(mor, mor, 127, 255, THRESH_BINARY | THRESH_OTSU);
    //Determine_Line_Info(mor, 0);

    threshold(src, src, 127, 255, THRESH_BINARY | THRESH_OTSU);
    Morph(src, mor);
    Find_Screw_Thread(mor);
    //Find_Screw_Thread(src_r90);

    // Convert image to binary
    threshold(mor, bw, 127, 255, THRESH_BINARY | THRESH_OTSU);


    int top, bottom, left, right;
    int borderType = BORDER_CONSTANT;
    // Initialize arguments for the filter
    top = (int) (0.1*bw.rows);
    bottom = top;
    left = (int) (0.1*bw.cols);
    right = left;

    //Mat dst;
    //Scalar border_color(255, 255, 255);
    //copyMakeBorder(bw, dst, top, bottom, left, right, borderType, border_color);
    //imshow("Border", dst );

    waitKey();
    return 0;
}
//=======================================================
