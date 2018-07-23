#pragma once
#include "stdafx.h"
#include <cv.h>

using namespace cv;

void inpaint(InputArray src, InputArray inpaintMask,
	OutputArray dst, double inpaintRadius, int flags);