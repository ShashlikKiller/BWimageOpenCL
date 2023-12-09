#include <iostream>

#include <chrono>
#include <iostream>
#include <cstring>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <CL/cl2.hpp>
#include <CL/cl.h>
#include <CL/opencl.h>
#include <CL/opencl.hpp>
#include <CL/cl_platform.h>
#include <CL/cl_layer.h>

using namespace cv;
using namespace std;

cv::Mat MakeGrayByPixel(cv::Mat image);
cv::Mat MakeGrayIGuess(cv::Mat image);
cv::Mat MakeGrayPtr(cv::Mat image);


int main(int argc, char* argv[])
{
    const char* filename_example = "C:\\Программы\\Картинки\\1.jpg";
    const char* filename = (argc == 2) ? argv[1] : filename_example;
    // Загрузка изображения
    cv::Mat imageOrigin = cv::imread(filename);
    // Проверка, что изображение загружено успешно
    if (imageOrigin.empty()) 
    {
        std::cout << "Can't load this image." << std::endl;
        return -1;
    }
    cv::Mat image;
    image = imageOrigin.cv::Mat::clone();

    // Далее идет работа с клоном изображения, чтобы с оригинальным ничего не произошло
    // даже если мы явно присвоим изображения через image A = image B, то А будет просто ссылкой на В и при изменении В измениться и А.


    // Конвертация изображения в черно-белое
    //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    //cv::imshow("cv", image);

    //image = MakeGrayByPixel(image);
    cv::imshow("cv1", image);
    printf("First");
    image = MakeGrayPtr(image);
    //image = MakeGrayIGuess(image);
    cv::imshow("cv2", image);
    printf("Second");

    // Сохранение черно-белого изображения
    //cv::imwrite("output_image.jpg", image);
    waitKey();

    // освобождаем ресурсы
    image.cv::Mat::release();
    // удаляем окно
    cv::destroyWindow("cv");
}


// как засечь время:
//double t0 = (double)getTickCount();
// здесь находится ваш код ...
//elapsed = ((double)getTickCount() – t0) / getTickFrequency();
//


cv::Mat MakeGrayPtr(cv::Mat image)
{
    cv::Mat _img(image.rows, image.cols, CV_8UC1);
    uchar _imgPix;
    uchar R, G, B;
    for (int i = 0; i < image.rows; i++)
    {
        Vec3b* imageRow = image.ptr<Vec3b>(i);
        uchar* _imgRow = _img.ptr<uchar>(i);
        for (int j = 0; j < image.cols; j++)
        {
            _imgPix = 0.299 * imageRow[j][2] + 0.587 * imageRow[j][1] + 0.114 * imageRow[j][0];
            _imgRow[j] = _imgPix;
        }
    }
    return _img;
}


void imgInfo(cv::Mat image)
{
    printf("[i] channels:  %d\n", image.channels());
    printf("[i] pixel depth: %d bits\n", image.depth());
    cv::MatSize imgsize = image.size;
    int rows = imgsize[0]; // Кол-во пикселей по вертикали
    int columns = imgsize[1]; // По вертикали
    printf("[i] rows:  %d\n", rows);
    printf("[i] columns:  %d\n", columns);
}

cv::Mat MakeGrayIGuess(cv::Mat image)
{
    cv::Mat _image;
    double _minRange, _MaxRange;
    cv::Point _mLoc, _MLoc;
    //cv::minMaxLoc(image, &_minRange, &_MaxRange, &_mLoc, &_MLoc); ORIGINAL в  чем ошибка хз
    try
    {
        cv::minMaxLoc(image, &_minRange, &_MaxRange, &_mLoc, &_MLoc);
        image.convertTo(_image, CV_8U, 255.0 / (_MaxRange - _minRange), -255 / _minRange);
        return _image;
    }
    catch (cv::Exception cvExp)
    {
        printf("Exception: %d", cvExp);
    }
    return image;
}

cv::Mat MakeGrayByPixel(cv::Mat image)
{
    MatSize _imgsize = image.size;
    int _rows = _imgsize[0]; // Кол-во пикселей по вертикали
    int _columns = _imgsize[1]; // По вертикали
    for (int x = 0; x < _rows; x++)
    {
        for (int y = 0; y < _columns; y++)
        {
            cv::Vec3b pix = image.at<cv::Vec3b>(x, y);
            float gray = 0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2];
            image.at<cv::Vec3b>(x, y) = cv::Vec3b(gray, gray, gray);
        }
    }
    return image;
}

int** GetImgPixMatrix(cv::Mat image)
{
    int** _pixMat = new int* [image.rows];
    for (int x = 0; x < image.rows; x++) 
    {
        _pixMat[x] = new int[image.cols];
        for (int y = 0; y < image.cols; y++) 
        {
            _pixMat[x][y] = image.at<uchar>(x, y);
        }
    }
    return _pixMat;
}
