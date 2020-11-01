#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <clocale>
#include <vector>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

template <class T>
T clamp(T v, int max, int min) {
	if (v > max)
		return max;

	else if (v < min)
		return min;

	return v;
}

Mat AddGaussianNoise(const Mat mSrc, double Mean = 0.0, double StdDev = 30.0)
{
	Mat mDst(mSrc.size(), mSrc.type());

	Mat mGaussian_noise = Mat(mSrc.size(), CV_16SC3);
	randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

	for (int Rows = 0; Rows < mSrc.rows; Rows++)
	{
		for (int Cols = 0; Cols < mSrc.cols; Cols++)
		{
			Vec3b Source_Pixel = mSrc.at<Vec3b>(Rows, Cols);
			Vec3b &Des_Pixel = mDst.at<Vec3b>(Rows, Cols);
			Vec3s Noise_Pixel = mGaussian_noise.at<Vec3s>(Rows, Cols);

			for (int i = 0; i < 3; i++)
			{
				int Dest_Pixel = Source_Pixel.val[i] + Noise_Pixel.val[i];
				Des_Pixel.val[i] = clamp<int>(Dest_Pixel, 255, 0);
			}
		}
	}

	return mDst;
}

float calculateNewPC(Mat photo, int x, int y, int rgb, int radius, int sigma) {
	float returnPC = 0;
	int size = 2 * radius + 1;
	float *vector = new float[size*size];
	float norm = 0;
	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++) {
			int idx = (i + radius)*size + j + radius;
			vector[idx] = exp(-(i*i + j * j) / sigma * sigma);
			norm += vector[idx];
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			vector[i*size + j] /= norm;
		}
	}
	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++) {
			int idx = (i + radius)*size + j + radius;
			returnPC += photo.at<Vec3b>(clamp<int>(x + j, photo.rows - 1, 0), clamp<int>(y + i, photo.cols - 1, 0))[rgb] * vector[idx];
		}
	}
	return returnPC;
}


Mat Gaussian_blur_filter(const Mat &photo, int radius, int sigma) {
	Mat result_Img;
	photo.copyTo(result_Img);
	int x, y;
	for (int x = 0; x < photo.rows; x++) {
		for (int y = 0; y < photo.cols; y++) {
			result_Img.at<Vec3b>(x, y)[0] = calculateNewPC(photo, x, y, 0, radius, sigma);//b
			result_Img.at<Vec3b>(x, y)[1] = calculateNewPC(photo, x, y, 1, radius, sigma);//g
			result_Img.at<Vec3b>(x, y)[2] = calculateNewPC(photo, x, y, 2, radius, sigma);//r
		}
	}
	return result_Img;
}

Mat Median_filter(const Mat &photo) {
	int r = 1;
	Mat result_Img;
	int size = 2 * r + 1, tmpR, tmpB, tmpG;
	int *colorR = new int[9];
	int *colorG = new int[9];
	int *colorB = new int[9];
	photo.copyTo(result_Img);
	for (int x = 0; x < photo.rows; x++)
	{
		for (int y = 0; y < photo.cols; y++)
		{
			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					int idx = (i + r)*size + j + r;

					colorR[idx] = photo.at<Vec3b>(clamp<int>(j + x, photo.rows - 1, 0), clamp<int>(i + y, photo.cols - 1, 0))[2];
					colorG[idx] = photo.at<Vec3b>(clamp<int>(j + x, photo.rows - 1, 0), clamp<int>(i + y, photo.cols - 1, 0))[1];
					colorB[idx] = photo.at<Vec3b>(clamp<int>(j + x, photo.rows - 1, 0), clamp<int>(i + y, photo.cols - 1, 0))[0];


				}
			}
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					int res = colorR[j] + colorG[j] + colorB[j],
						res1 = colorR[j + 1] + colorG[j + 1] + colorB[j + 1];
					if (res > res1) {
						tmpR = colorR[j];
						tmpG = colorG[j];
						tmpB = colorB[j];
						colorR[j] = colorR[j + 1];
						colorG[j] = colorG[j + 1];
						colorB[j] = colorB[j + 1];
						colorR[j + 1] = tmpR;
						colorG[j + 1] = tmpG;
						colorB[j + 1] = tmpB;
					}
				}
			}
			result_Img.at<Vec3b>(x, y)[0] = colorB[4];
			result_Img.at<Vec3b>(x, y)[1] = colorG[4];
			result_Img.at<Vec3b>(x, y)[2] = colorR[4];
		}
	}
	return result_Img;
}

int main(int argc, char *argv[]) {
	setlocale(LC_ALL, "Rus");
	Mat main_Image = imread("first_image.jpg");
	if (main_Image.empty()) {
		cout << "Error: the image has been incorrectly loaded." << endl;
		system("pause");
		return 0;
	}
	double bright = 0.0, coef = 50.0;

	namedWindow("DEFAULT_PICTURES");
	imshow("DEFAULT_PICTURES", main_Image);
	waitKey(0);
	cvDestroyWindow("DEFAULT_PICTURES");

	namedWindow("MODELING_GAUS_NOISE");
	Mat res;
	main_Image.copyTo(res);
	long double t1_my_gaus_noise = clock();
	Mat image_with_noise = AddGaussianNoise(res, bright, coef);
	long double t2_my_gaus_noise = clock();
	t2_my_gaus_noise -= t1_my_gaus_noise;
	cout << "Clock_of_MODELING_GAUS_NOISE: " << setprecision(15) << t2_my_gaus_noise / CLOCKS_PER_SEC << endl;
	imshow("MODELING_GAUS_NOISE", image_with_noise);
	imwrite("D:gaus_noise.jpg", image_with_noise);
	waitKey(0);
	cvDestroyWindow("MODELING_GAUS_NOISE");

	namedWindow("MY_GAUS_FILTER");
	Mat res_1;
	image_with_noise.copyTo(res_1);
	long double t1_myGaus = clock();
	Mat image_without_noise_2 = Gaussian_blur_filter(res_1, 3, 10);
	long double t2_myGaus = clock();
	t2_myGaus -= t1_myGaus;
	cout << "Clock_of_MY_GAUS_FILTER: " << setprecision(15) << t2_myGaus / CLOCKS_PER_SEC << endl;
	imshow("MY_GAUS_FILTER", image_without_noise_2);
	imwrite("D:my_gaus_filter.jpg", image_without_noise_2);
	waitKey(0);
	cvDestroyWindow("MY_GAUS_FILTER");

	namedWindow("MY_MEDIAN_FILTER");
	Mat res_4;
	image_with_noise.copyTo(res_4);
	long double t1_myMedian = clock();
	Mat image_without_noise_3 = Median_filter(res_4);
	long double t2_myMedian = clock();
	t2_myMedian -= t1_myMedian;
	cout << "Clock_of_MY_MEDIAN_FILTER: " << setprecision(15) << t2_myMedian / CLOCKS_PER_SEC << endl;
	imshow("MY_MEDIAN_FILTER", image_without_noise_3);
	imwrite("D:my_median_filter.jpg", image_without_noise_3);
	waitKey(0);
	cvDestroyWindow("MY_MEDIAN_FILTER");

	namedWindow("OPENCV_MEDIAN");
	Mat res_2;
	Mat image_without_noise;
	image_with_noise.copyTo(res_2);
	long double t1_opencv_median = clock();
	medianBlur(res_2, image_without_noise, 7);
	long double t2_opencv_median = clock();
	t2_opencv_median -= t1_opencv_median;
	cout << "Clock_of_OPENCV_MEDIAN: " << setprecision(15) << t2_opencv_median / CLOCKS_PER_SEC << endl;
	imshow("OPENCV_MEDIAN", image_without_noise);
	imwrite("D:opencv_median_filter.jpg", image_without_noise);
	waitKey(0);
	cvDestroyWindow("OPENCV_MEDIAN");

	namedWindow("OPENCV_GAUS");
	Mat res_3;
	Mat image_without_noise_1;
	image_with_noise.copyTo(res_3);
	long double t1_opencv_gaus = clock();
	GaussianBlur(res_3, image_without_noise_1, Size(5, 5), 0, 0);
	long double t2_opencv_gaus = clock();
	t2_opencv_gaus -= t1_opencv_gaus;
	cout << "Clock_of_OPENCV_GAUS: " << setprecision(15) << t2_opencv_gaus / CLOCKS_PER_SEC << endl;
	imshow("OPENCV_GAUS", image_without_noise_1);
	imwrite("D:opencv_gaus_filter.jpg", image_without_noise_1);
	waitKey(0);
	cvDestroyWindow("OPENCV_GAUS");

	system("pause");
	return 0;
}
