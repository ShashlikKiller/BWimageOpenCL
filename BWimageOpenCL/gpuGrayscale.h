#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "arraysConversion.h"
#include <iostream>
#include <opencv2/opencv.hpp>


#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#include <CL/cl.h>
#include <CL/opencl.h>
#include <CL/opencl.hpp>
#include <CL/cl_platform.h>
#include <CL/cl_layer.h>


using namespace cv;
using namespace std;

void GrayscaleKernel(int*** pixMatrix);
void KernelExecution(cl::Device device, int*** PixMatrix);
int* GetGrayscaledImage();
double GetElapsed();