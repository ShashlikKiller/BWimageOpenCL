#pragma once
#include <stdexcept>
#include <iostream>
#include <opencv2/opencv.hpp>
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#include <CL/cl.h>
#include <CL/opencl.h>
#include <CL/opencl.hpp>
#include <CL/cl_platform.h>
#include <CL/cl_layer.h>
#include <ppl.h>

using namespace cv;
using namespace Concurrency;
using namespace std;

int* convertTo1D(int rows, int cols, int*** pixMatrix);
int** convertTo2D(int* resultImageMatrix, int width, int height);
int*** convertTo3D(cv::Vec3b** PixMatrix);
cv::Mat GetGrayscaledImage(int** PixMatrix);
cv::Vec3b** GetImgPixChannelMatrix(cv::Mat image);

