#include "FeatureDetector.h"

void Blob::detectHarris(const Mat& src, Mat& dest, float alpha, float thresh) {
	// Step1: Grayscaling source image to get a grayscale image
	Mat srcGray;
	cvtColor(src, srcGray, COLOR_BGR2GRAY);
	dest = Mat(src.size(), CV_32FC1);

	// Step2: Apply Gaussian filter to reduce noise
		// Size (11,11) is kWidth = 11, kHeight = 11, sigma = 1.0
	GaussianBlur(srcGray, srcGray, Size(3, 3), 1.0, 0.0, BORDER_DEFAULT);

	// Step3: Compute magnitude of the x and y gradients at each pixel by applying Sobel algorithm
	Mat gradientX, gradientY, x2y2Mat, xyMat, traceMat;
	vector<float> kernelX; // Kx
	vector<float> kernelY; // Ky
		/*
			|1  0  -1|
			|2  0  -2| =  Kx
			|1  0  -1|
		*/
	kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);
	kernelX.push_back(2); kernelX.push_back(0); kernelX.push_back(-2);
	kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);

	/*
		|-1  -2  -1|
		| 0   0   0| =  Ky
		| 1   2   1|
	*/

	kernelY.push_back(-1); kernelY.push_back(-2); kernelY.push_back(-1);
	kernelY.push_back(0); kernelY.push_back(0); kernelY.push_back(0);
	kernelY.push_back(1); kernelY.push_back(2); kernelY.push_back(1);


	Convolution convolution;
	convolution.SetKernel(kernelX, 3, 3);
	convolution.DoConvolution(srcGray, gradientX);

	convolution.SetKernel(kernelY, 3, 3);
	convolution.DoConvolution(srcGray, gradientY);

	// Step3: Construct M 
	Mat Ix2, Iy2, Ixy;
	pow(gradientX, 2.0, Ix2); // Ix2 = gradX^2
	pow(gradientY, 2.0, Iy2); // Iy2 = grady^2	
	multiply(gradientX, gradientY, Ixy);

	GaussianBlur(Ix2, Ix2, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(Iy2, Iy2, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(Ixy, Ixy, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);

	// Step4: Compute R = detM - alpha * traceM^2
	multiply(Ix2, Iy2, x2y2Mat);
	multiply(Ixy, Ixy, xyMat);
	pow((Ix2 + Iy2), 2.0, traceMat);

	Mat R = (x2y2Mat - xyMat) - alpha * traceMat;

	// Step5: Retain local maxima
	Mat R_norm;
	normalize(R, R_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	// Step6: Compare with Thresh
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((int)R_norm.at<float>(i, j) > thresh)
			{
				circle(src, Point(j, i), 10, Scalar(255, 0, 0), 1, 8, 0);
			}
		}
	}
	dest = src;
}

void Blob::detectBlob_LoG(const Mat& src, Mat& dest, int thresh) {
	// Step1: convert to gray image
	Mat grayImage;
	cvtColor(src, grayImage, COLOR_BGR2GRAY);

	// Step2: Apply Gaussian filter
	GaussianBlur(grayImage, grayImage, Size(7, 7), 0.0, 0.0, BORDER_DEFAULT);

	// Step3: Apply Laplacian to get LoG
	Laplacian(grayImage, grayImage, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);

	// Step4: Normalize(MinMax) to get localMax
	Mat R_norm;
	normalize(grayImage, R_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// Step5: Compare with Thresh
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((int)R_norm.at<float>(i, j) > thresh)
			{
				circle(src, Point(j, i), 10, Scalar(255, 0, 0), 1, 8, 0);
			}
		}
	}
	dest = src;
}

void Blob::detectBlob_DoG(const Mat& src, Mat& dest, int thresh) {
	// Step1: Convert to gray image
	Mat grayImage, Low_Sig_Gauss, High_Sig_Gauss;
	cvtColor(src, grayImage, COLOR_BGR2GRAY);

	// Step2: Apply low threshold and high threshold Gaussian
	GaussianBlur(grayImage, Low_Sig_Gauss, Size(3, 3), 0.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(grayImage, High_Sig_Gauss, Size(7, 7), 0.0, 0.0, BORDER_DEFAULT);

	// Step3: Calculate Differrence of Gaussian
	grayImage = Low_Sig_Gauss - High_Sig_Gauss;

	// Step4: Normalize(MinMax) to get localMax
	Mat R_norm;
	normalize(grayImage, R_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// Step5: Compare with Thresh
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((int)R_norm.at<float>(i, j) > thresh)
			{
				circle(src, Point(j, i), 10, Scalar(255, 0, 0), 1, 8, 0);
			}
		}
	}
	dest = src;
}

void Blob::detectBlob_DoL(const Mat& src, Mat& dest, int thresh) {
	//convert to gray image and remove noise
	Mat grayImage, Low_Sig_Gauss, High_Sig_Gauss;
	cvtColor(src, grayImage, COLOR_BGR2GRAY);
	GaussianBlur(grayImage, Low_Sig_Gauss, Size(3, 3), 0.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(grayImage, High_Sig_Gauss, Size(7, 7), 0.0, 0.0, BORDER_DEFAULT);

	Laplacian(Low_Sig_Gauss, Low_Sig_Gauss, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);
	Laplacian(High_Sig_Gauss, High_Sig_Gauss, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);

	grayImage = Low_Sig_Gauss - High_Sig_Gauss;
	Mat R_norm;
	normalize(grayImage, R_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// Step6: Compare with Thresh
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((int)R_norm.at<float>(i, j) > thresh)
			{
				circle(src, Point(j, i), 10, Scalar(255, 0, 0), 1, 8, 0);
			}
		}
	}
	dest = src;
}

void Blob::detectBlob_LoG(const Mat& src, Mat& dest, int thresh, vector<KeyPoint>& keyvector) {
	// Step1: convert to gray image
	Mat grayImage;
	cvtColor(src, grayImage, COLOR_BGR2GRAY);

	// Step2: Apply Gaussian filter
	GaussianBlur(grayImage, grayImage, Size(7, 7), 0.0, 0.0, BORDER_DEFAULT);

	// Step3: Apply Laplacian to get LoG
	Laplacian(grayImage, grayImage, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);

	// Step4: Normalize(MinMax) to get localMax
	Mat R_norm;
	normalize(grayImage, R_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// Step5: Compare with Thresh
	vector<Point2f> vecPoint;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((int)R_norm.at<float>(i, j) > thresh)
			{
				vecPoint.push_back(Point2f(j * 1.f, i * 1.f));
			}
		}
	}

	for (size_t i = 0; i < vecPoint.size(); i++) {
		keyvector.push_back(cv::KeyPoint(vecPoint[i], 1.f));
	}
	dest = src;
}

