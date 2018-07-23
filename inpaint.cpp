#include "stdafx.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, _TCHAR* argv[])
{
	Mat img0 = imread("test1.jpg", -1);  /*������޸�ͼ�� ����2�������У�>0 Return a 3-channel color image.��ɫͼ
																	=0 Return a grayscale image.�Ҷ�ͼ��
																	<0 Return the loaded image as is (with alpha channel).Ϊ8bit�Ĳ�ɫͼ��
									 */
	namedWindow("image", 1);
	Mat img = img0.clone();
	imshow("image", img);

	Mat inpaintMask = imread("5.jpg", 0);   //�������Ϊ���أ�255��255��255������ɫ�����޸�����Ϊ��0��0��0������ɫ���Ĳ����޸�����
	imshow("mask", inpaintMask);

	Mat inpainted = img.clone();
	Mat inpainted1 = img.clone();

	imgInpaint(img, inpaintMask, inpainted, 11);   //����д��FMM�޸��㷨��11�����޸��뾶
	imshow("inpainted image", inpainted);
	
//	inpaint(img, inpaintMask, inpainted1, 15, 1);  //opencv�Դ���FMM�޸��㷨
//	imshow("cv inpainted image", inpainted1);


	cout << "image inpainted using FMM!" << endl;

	waitKey();

	return 0;
}