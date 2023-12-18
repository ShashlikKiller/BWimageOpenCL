#include <iostream>
#include <opencv2/opencv.hpp>

#include "arraysConversion.h"
#include "cpuGrayscale.h"
#include "gpuGrayscale.h"

using namespace cv;
using namespace std;

double time_start, time_cl, time_cvtColor, time_ptr, time_ptrParallelfor, time_getimagecl, time_bypixel;
int ROWS, COLS;

int main(int argc, char* argv[])
{
    // Загрузка изображения:
    std::string path = __FILE__;
    size_t pos = path.find_last_of('/');
    std::string filename_example = path.substr(0, pos + 1) + "example.jpg";
    std::string filename = (argc == 2) ? argv[1] : filename_example;
    cv::Mat imageOrigin = cv::imread(filename);
    if (imageOrigin.empty()) 
    {
        std::cout << "Can't load this image.\n";
        return -1;
    }
    // Далее идет работа с клоном изображения, чтобы с оригинальным ничего не произошло
    // даже если мы явно присвоим изображения через image A = image B, то А будет просто ссылкой на В и при изменении В измениться и А.
    cv::Mat image = imageOrigin.cv::Mat::clone();
    cv::Mat image_cvtColor;
    // Преобразование изображения в массив из значений цветных каналов для работы над ним в ядре
    Vec3b** PixMatrix = GetImgPixChannelMatrix(image);
    int*** PixMatrix3D = convertTo3D(PixMatrix);
    ROWS = image.size[0];
    COLS = image.size[1];
    GrayscaleKernel(PixMatrix3D);
    int** GrayPixMatrix = convertTo2D(GetGrayscaledImage(), COLS, ROWS);

    //Making an image from CL matrix:
    time_start = (double)getTickCount();
    cv::Mat image_cl = GetGrayscaledImage(GrayPixMatrix);
    time_getimagecl = ((double)getTickCount() - time_start) / getTickFrequency();

    //MakeGrayPtrParallel_for
    time_start = (double)getTickCount();
    cv::Mat image_makeGrayPtrParallel_for = MakeGrayPtrParallel_for(image);
    time_ptrParallelfor = ((double)getTickCount() - time_start) / getTickFrequency();

    //MakeGrayPtr
    time_start = (double)getTickCount();
    cv::Mat image_makeGrayPtr = MakeGrayPtr(image);
    time_ptr = ((double)getTickCount() - time_start) / getTickFrequency();

    //MakeGrayByPixel
    time_start = (double)getTickCount();
    cv::Mat image_makeGrayByPixel = MakeGrayByPixel(image);
    time_bypixel = ((double)getTickCount() - time_start) / getTickFrequency();

    //cv::cvtColor
    time_start = (double)getTickCount();
    cv::cvtColor(image, image_cvtColor, cv::COLOR_BGR2GRAY);
    time_cvtColor = ((double)getTickCount() - time_start) / getTickFrequency();

    // Показ готовых изображений:
    cv::imshow("cv_cvtColor", image_cvtColor);
    cv::imshow("cv_parallelFor", image_makeGrayPtrParallel_for);
    cv::imshow("cl_kernel",image_cl);

    std::cout << "Time for kernel imageProcessing: " << GetElapsed() << " seconds.\n";
    std::cout << "Time for building the image from CL matrix: " << time_getimagecl << " seconds.\n";
    std::cout << "Time for ptrParallel_for: " << time_ptrParallelfor << " seconds.\n";
    std::cout << "Time for ptr: " << time_ptr << " seconds.\n";
    std::cout << "Time for byPixel: " << time_bypixel << " seconds.\n";
    std::cout << "Time for cv::cvtColor: " << time_cvtColor << " seconds.\n";

    waitKey();

    // освобождаем ресурсы
    image.cv::Mat::release();
    imageOrigin.cv::Mat::release();
    image_makeGrayByPixel.cv::Mat::release();
    image_cvtColor.cv::Mat::release();
    image_makeGrayPtr.cv::Mat::release();
    image_cl.cv::Mat::release();
    image_makeGrayPtrParallel_for.cv::Mat::release();
    
    // Удаление окон с изображениями:
    cv::destroyWindow("cv_cvtColor");
    cv::destroyWindow("cv_parallelFor");
    cv::destroyWindow("cl_kernel");
}

cv::Mat GetImgFromPixMatrix(cv::Vec3b** PixMatrix)
{
    cv::Mat imageResult(ROWS, COLS, CV_8UC3);
    for (int x = 0; x < ROWS; x++)
    {
        for (int y = 0; y < COLS; y++)
        {
            imageResult.at<cv::Vec3b>(x, y) = PixMatrix[x][y];
        }
    }
    return imageResult;
}