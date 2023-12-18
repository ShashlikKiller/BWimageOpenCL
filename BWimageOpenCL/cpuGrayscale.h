#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <ppl.h>


cv::Mat MakeGrayPtr(cv::Mat image);
cv::Mat MakeGrayPtrParallel_for(cv::Mat image);
cv::Mat MakeGrayByPixel(cv::Mat image);
float GetGrayPix(int R, int G, int B);

