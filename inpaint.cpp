#include "stdafx.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, _TCHAR* argv[])
{
	Mat img0 = imread("test1.jpg", -1);  /*读入待修复图像 ，第2个参数有：>0 Return a 3-channel color image.彩色图
																	=0 Return a grayscale image.灰度图像
																	<0 Return the loaded image as is (with alpha channel).为8bit的彩色图像
									 */
	namedWindow("image", 1);
	Mat img = img0.clone();
	imshow("image", img);

	Mat inpaintMask = imread("5.jpg", 0);   //掩码矩阵，为像素（255，255，255）（白色）作修复处理，为（0，0，0）（黑色）的不作修复处理
	imshow("mask", inpaintMask);

	Mat inpainted = img.clone();
	Mat inpainted1 = img.clone();

	imgInpaint(img, inpaintMask, inpainted, 11);   //重新写的FMM修复算法，11——修复半径
	imshow("inpainted image", inpainted);
	
//	inpaint(img, inpaintMask, inpainted1, 15, 1);  //opencv自带的FMM修复算法
//	imshow("cv inpainted image", inpainted1);


	cout << "image inpainted using FMM!" << endl;

	waitKey();

	return 0;
}