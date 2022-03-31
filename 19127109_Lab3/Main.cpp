#include <iostream>
#include "MatchBySIFT_KNN.h"
#include <opencv2/highgui.hpp>
#include <string>
using namespace cv;
using namespace std;


int main(int argc, char* argv[]) {
    if (argc == 5) {
        string op, in, out;
        op = argv[1]; in = argv[3]; out = argv[4];
        int thresh = stoi(argv[2]);
        if (op == "detectHarris") {
            // Read input image
            Mat src = imread(in);
            // Check if input is empty
            if (src.empty())
            {
                cout << "Could not open or find the image!\n" << endl;
                return -1;
            }
            Blob b;
            Mat dst;
            b.detectHarris(src, dst, 0.06, thresh);
            imwrite(out, dst);
            return 0;
        }
        else if (op == "detectBlob") {
            // Read input image
            Mat src = imread(in);
            // Check if input is empty
            if (src.empty())
            {
                cout << "Could not open or find the image!\n" << endl;
                return -1;
            }
            Mat dst;
            Blob b;
            b.detectBlob_LoG(src, dst, thresh);
            imwrite(out, dst);
            return 0;
        }
        else if (op == "detectDOG") {
            // Read input image
            Mat src = imread(in);
            // Check if input is empty
            if (src.empty())
            {
                cout << "Could not open or find the image!\n" << endl;
                return -1;
            }
            Mat dst;
            Blob b;
            b.detectBlob_DoG(src, dst, thresh);
            imwrite(out, dst);
            return 0;
        }
        else if (op == "detectDOL") {
            // Read input image
            Mat src = imread(in);
            // Check if input is empty
            if (src.empty())
            {
                cout << "Could not open or find the image!\n" << endl;
                return -1;
            }
            Mat dst;
            Blob b;
            b.detectBlob_DoL(src, dst, thresh);
            imwrite(out, dst);
            return 0;
        }
    }
    // .exe matchBySIFT in1 in2 out ratio threshold
    else if (argc == 7) {
        string op, in1, in2, out;
        op = argv[1]; 
        in1 = argv[2];
        in2 = argv[3];
        out = argv[4];
        float ratio = stof(argv[5]);
        int thresh = stoi(argv[6]);
        if (op == "matchBySIFT") {
            // Read input image
            Mat src1 = imread(in1), src2 = imread(in2);
            // Check if input is empty
            if (src1.empty() || src2.empty())
            {
                cout << "Could not open or find the image!\n" << endl;
                return -1;
            }
            MatchSift s;
            Mat dst;
            s.matchBySIFT(src1, src2, dst, ratio, thresh); 
            imwrite(out, dst);
            return 0;
        }
    }
}

//int main() {
//    string path = "D:\\training_images\\";
//
//    Mat src1 = imread("D:\\training_images\\Rengoku.jpg"),
//        src2 = imread("D:\\training_images\\Ace.jpg");
//
//    MatchSift s;
//    Mat dst;
//    s.matchBySIFT(src1, src2, dst, 0.8f, 100);
//    
//    imwrite("Donut.jpg", dst);
//
//    return 0;
//
//    
//}


