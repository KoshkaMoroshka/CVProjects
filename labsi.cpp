#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;


//-------------lab1---------------------
Mat drawhist(Mat& src, Mat& dst, int w, int h, int size)
{
	vector<Mat> bgr;
	split(src, bgr);
	Mat hist;
	float range[] = { 0,256 };
	const float* ranges = { range };
	calcHist(&bgr[0], 1, 0, Mat(), hist, 1, &size, &ranges);

	normalize(hist, hist, 0, h, NORM_MINMAX, -1, Mat());
	int x = 0, y0 = h;
	Scalar col(127, 127, 127);
	for (int i = 0; i < size;)
	{
		line(dst, Point(x, y0), Point(x, y0 - (int)hist.at<float>(i)), col);
		x = (int)((float)++i / size * w);
	}
	return hist;
}

//-------------lab2---------------------
int otsu(Mat hst, int i1, int i2)
{
	int size = i2 - i1;
	float area = 0;
	for (int i = 0; i < size; i++)
		area += hst.at<float>(i1 + i);
	float divn = 1.0f / area;

	float q = 0;
	float* q1 = new float[size];
	for (int i = 0; i < size; i++)
		q1[i] = q += hst.at<float>(i1 + i) * divn;
	for (int i = 0; i < size; i++)
		q1[i] /= q;
	q = 1.0f;

	float m = 0;
	float* m1 = new float[size];
	for (int i = 0; i < size; i++)
		m1[i] = m += hst.at<float>(i1 + i) * divn * (1 + i);
	for (int i = 0; i < size; i++)
		m1[i] /= m;
	m = 1.0f;

	int imax = 0;
	float tmax = 0;
	for (int i = 0; i < size; i++)
	{
		float _q1 = q1[i], _q2 = q - _q1;
		float _m1 = m1[i], _m2 = m - _m1;
		float t = _m1 / _q1 - _m2 / _q2;
		t *= t * _q1 * _q2;
		if (_q1 * _q2 == 0)
			t = 0;

		if (t <= tmax)
			continue;
		imax = i;
		tmax = t;
	}
	delete[] q1, m1;
	return i1 + imax;
}

//-------------lab4---------------------
template<class ty> void to8U(Mat src, Mat dst)
{
	ty min = src.at<ty>(0, 0), max = min;
	for (int y = 1; y < src.rows; y++)
		for (int x = 1; x < src.cols; x++)
		{
			ty t = src.at<ty>(y, x);
			min = t < min ? t : min;
			max = t > max ? t : max;
		}
	float k = (float)0xFF / (max - min);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			ty t = src.at<ty>(y, x);
			uchar r = (uchar)((t - min) * k);
			dst.at<uchar>(y, x) = r;
		}
	return;
}

//-------------lab6---------------------
int paint(Mat src, uchar col)
{
	uchar c = 2;
	for (int y = 1; y < src.rows - 1; y++)
		for (int x = 1; x < src.cols - 1; x++)
		{
			uchar t = src.at<uchar>(y, x);
			if (t != col)
				continue;
			floodFill(src, Point(x, y), c++);
		}
	return c;
}

//-------------yeet---------------------
Mat cell(int r1, int r2)
{
	float s = 0;
	Mat res(r1 << 1, r1 << 1, CV_32F);
	int R1 = r1 * r1, R2 = r2 * r2;
	for (int y = -r1; y < r1; y++)
		for (int x = -r1; x < r1; x++)
		{
			int r = x * x + y * y;
			s += res.at<float>(y + r1, x + r1) = (r >= R2 && r < R1);
		}
	s = 1.0f / s;
	for (int y = 0; y < (r1 << 1); y++)
		for (int x = 0; x < (r1 << 1); x++)
			float t = res.at<float>(y, x) *= s;
	return res;
}
void localmax(Mat src, Mat dst, int r, uchar e = 0)
{
	Mat m(r << 1, r << 1, CV_8U, Scalar(1));
	m.at<uchar>(r, r) = 0;
	dilate(src, dst, m);
	dst = dst - src <= e;
	return;
}
void yeet(void)
{
	int type = CV_8UC1;
	Mat src, src3c = imread("spinmeround.png");
	cvtColor(src3c, src, COLOR_BGR2GRAY);
	Size size = src.size();
	Mat inv;
	bitwise_not(src, inv);
	imshow("src", inv);
	
	//----------------background------------------
	int w = 256, h = 200, c = 256;
	Mat hist1(h, w, type, Scalar(0));
	Mat hst1 = drawhist(inv, hist1, w, h, c);

	int i0 = 0;
	uchar m = hst1.at<int>(0);
	for (int i = 1; i < c; i++)
	{
		uchar t = hst1.at<int>(i);
		if (t < m)
			continue;
		m = t;
		i0 = i;
	}
	int i2 = otsu(hst1, i0, c);

	line(hist1, Point(i2, h), Point(i2, 0), 0xFF);
	imshow("hist 1", hist1);

	Mat m1(size, type);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			uchar t = inv.at<uchar>(y, x);
			m1.at<uchar>(y, x) = t >= i2 ? 0xFF : 0;
		}
	imshow("1 background", m1);
	Mat mask;
	bitwise_not(m1, mask);

	//--------------cell--response------------------
	Mat cel = cell(10, 9);
	Mat cel8u(cel.rows, w, type);
	to8U<float>(cel, cel8u);
	imshow("cell", cel8u);

	Mat m2(size, type);
	filter2D(m1, m2, 1, cel);
	to8U<schar>(m2, m1);
	imshow("2 cell response", m1);


	//----------------kek------------------
	Mat m4(size, type);
	localmax(m1, m4, 32, 0);
	imshow("3 local max", m4);

	int r1 = 12;
	Mat m5, ker1 = getStructuringElement(MORPH_ELLIPSE, Size(r1, r1));
	erode(mask, m5, ker1);
	floodFill(m5, Point(0, 0), 0);
	imshow("mask", m5);

	bitwise_and(m4, m5, m4);
	imshow("4 mask", m4);

	int r2 = 24, r3 = 12;
	Mat ker2 = getStructuringElement(MORPH_ELLIPSE, Size(r2, r2));
	Mat ker3 = getStructuringElement(MORPH_ELLIPSE, Size(r3, r3));
	dilate(m4, m2, ker2);
	dilate(m4, m5, ker3); // nuclears
	bitwise_not(m5, m5);
	bitwise_and(m2, m5, m4);

	bitwise_not(m5, m5);
	int n = paint(m5, 0xFF);
	to8U<uchar>(m5, m5);
	imshow("5 delta dilate", m4);
	imshow("res: " + to_string(n - 1), m5);


	waitKey(0);
	return;
}
int main()
{
	yeet();
	return 0;
}
