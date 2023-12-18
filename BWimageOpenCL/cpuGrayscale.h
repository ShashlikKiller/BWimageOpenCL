#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <ppl.h>


cv::Mat getGrayPtr(cv::Mat image);
cv::Mat getGrayPtrParallel_for(cv::Mat image);
cv::Mat getGrayByEveryPixel(cv::Mat image);
float getGrayscaledPixel(int R, int G, int B);

