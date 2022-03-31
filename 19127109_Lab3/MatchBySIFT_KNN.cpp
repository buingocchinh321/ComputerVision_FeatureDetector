#include "MatchBySIFT_KNN.h"

void MatchSift::matchBySIFT(const Mat& img1, const Mat& img2, Mat& dst, float ratio, int threshold) {

	// Use class SiftFeatureDetector to compute descriptor from keypoints
	cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
	Blob b;
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Step1: Detect Keypoints and compute Descriptors
	b.detectBlob_LoG(img1, dst, threshold, keypoints1);
	b.detectBlob_LoG(img2, dst, threshold, keypoints2);

	detector->compute(img1, keypoints1, descriptors1);
	detector->compute(img2, keypoints2, descriptors2);

	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// Since SIFT is a floating-point descriptor NORM_L2 is used
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	//-- Filter matches using the Lowe's ratio test
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	//-- Draw matches
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	dst = img_matches;
}