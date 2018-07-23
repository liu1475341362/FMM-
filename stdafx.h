#pragma once

#include "targetver.h"
#include <cv.h>
#include <stdio.h>
#include <tchar.h>
#include <iostream>

void imgInpaint(cv::InputArray src, cv::InputArray inpaintMask,
	cv::OutputArray dst, double inpaintRadius);
