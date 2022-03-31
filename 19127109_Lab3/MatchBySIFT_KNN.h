#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "FeatureDetector.h"
#ifndef MatchBySIFT_KNN_H_
#define MatchBySIFT_KNN_H_
using namespace cv::ml;
#endif // !MatchBySIFT_KNN_H_
#pragma once
class MatchSift {
public:
	MatchSift() {}
	/*
	img1: Source image 1
	img2: Source image 2
	dst: Destination image
	ratio: Ratio threshold pf KNN Matching
	threshold: Threshold LOG
	*/
	void matchBySIFT(const Mat& img1, const Mat& img2, Mat& dst, float ratio, int threshold);
};
