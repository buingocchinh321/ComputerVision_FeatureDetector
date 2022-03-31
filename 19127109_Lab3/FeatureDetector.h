#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "Convolution.h"
#ifndef FeatureDetector_H_
#define FeatureDetector_H_
using namespace cv;
using namespace std;
#endif // !FeatureDetector_H_
#pragma once

class Blob {
public:
	Blob() {

	}

	void detectHarris(const Mat& src, Mat& dest, float alpha, float thresh);

	void detectBlob_LoG(const Mat& src, Mat& dest, int thresh);

	void detectBlob_LoG(const Mat& src, Mat& dest, int thresh, vector<KeyPoint>& keyvector);

	void detectBlob_DoG(const Mat& src, Mat& dest, int thresh);

	void detectBlob_DoL(const Mat& src, Mat& dest, int thresh);

};


