#include <iostream>

#include <chrono>
#include <iostream>
#include <cstring>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#include <CL/cl.h>
#include <CL/opencl.h>
#include <CL/opencl.hpp>
#include <CL/cl_platform.h>
#include <CL/cl_layer.h>

//Из прошлой работы:
#define TESTS_NUMBER = 1
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <conio.h>
#include <thread>
#include <ppl.h>
#include <array>
#include <sstream>
#include <vector>

using namespace cv;
using namespace Concurrency;
using namespace std;

float GetGrayPix(int R, int G, int B);
cv::Mat MakeGrayPtrParallel_for(cv::Mat imageOrigin);
cv::Mat MakeGrayByPixel(cv::Mat image);
cv::Mat MakeGrayIGuess(cv::Mat image);
cv::Mat MakeGrayPtr(cv::Mat image);
int** GetImgPixMatrix(cv::Mat image);


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
    cv::Mat image, image2;
    image = imageOrigin.cv::Mat::clone();
    image2 = imageOrigin.cv::Mat::clone();

    // Далее идет работа с клоном изображения, чтобы с оригинальным ничего не произошло
    // даже если мы явно присвоим изображения через image A = image B, то А будет просто ссылкой на В и при изменении В измениться и А.


    // Конвертация изображения в черно-белое
    //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    //cv::imshow("cv", image);

    int** matrix = GetImgPixMatrix(image);

    //image2 = MakeGrayByPixel(image2);
    //cv::imshow("cv1", image2);
    //printf("First");
    //image = MakeGrayPtr(image);
    //image = MakeGrayIGuess(image);
    //cv::imshow("cv2", image);
    printf("Second");
    image = MakeGrayPtrParallel_for(image);
    cv::imshow("cv3", image);
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

cv::Mat MakeGrayPtr(cv::Mat image) // Added parralel_for
{
    __int64 start_count;
    __int64 end_count;
    __int64 freq;
    cv::Mat _img(image.rows, image.cols, CV_8UC1);
    uchar _imgPix;
    for (int x = 0; x < image.rows; x++)
    {
        Vec3b* imageRow = image.ptr<Vec3b>(x);
        uchar* _imgRow = _img.ptr<uchar>(x);
        for (int y = 0; y < image.cols; y++)
        {
            _imgPix = GetGrayPix(imageRow[y][2], imageRow[y][1], imageRow[y][0]);  // 0.299 * imageRow[y][2] + 0.587 * imageRow[y][1] + 0.114 * imageRow[y][0];
            _imgRow[y] = _imgPix;
        }
    }
    return _img;
}

cv::Mat MakeGrayPtrParallel_for(cv::Mat imageOrigin)
{
    cv::Mat imageGray(imageOrigin.rows, imageOrigin.cols, CV_8UC1);
    parallel_for(size_t(0), size_t(imageOrigin.rows), [&imageOrigin, &imageGray](size_t x) // добавляем image в список захвата        
        {
            uchar _imgPix;
            Vec3b* _imageOriginRow = imageOrigin.ptr<Vec3b>(x);
            uchar* _imgRow = imageGray.ptr<uchar>(x);
            for (int y = 0; y < imageOrigin.cols; y++)
            {
                _imgPix = GetGrayPix(_imageOriginRow[y][2], _imageOriginRow[y][1], _imageOriginRow[y][0]); // 0.299 * _imageOriginRow[y][2] + 0.587 * _imageOriginRow[y][1] + 0.114 * _imageOriginRow[y][0];
                _imgRow[y] = _imgPix;
            }
        });
    return imageGray;
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

float GetGrayPix(int R, int G, int B)
{
    return 0.299 * R + 0.587 * G + 0.114 * B;
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
            float gray = GetGrayPix(pix[0], pix[1], pix[2]); //0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2];
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

int** hostImage;
void PerformTestOnDevice(cl::Device device, cv::Mat image)
{
    //preparing:
    const int IMAGE_SIZE = image.size[0]*image.size[1];
    int** ResultImagePixelsMatrix; // ON DEVICE
    int** ImagePixelsMatrix; // ON HOST
    cv::Mat _imgResult(image.rows, image.cols, CV_8UC1);
    const int RESULT_IMAGE_SIZE = _imgResult.size[0] * _imgResult.size[1];
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl << endl;

    //For the selected device create a context
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);

    // Создаем очередь для девайса
    cl::CommandQueue queue(context, device);

    ////Clean output buffers
    //fill_n(ImagePixelsMatrix, image.size, 0);

    ////Create memory buffers
    //cl::Buffer ImagePixels_host = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(__int32), ImagePixelsMatrix);
    //cl::Buffer ImagePixels_device = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(__int32), ResultImagePixelsMatrix);
    fill_n(ResultImagePixelsMatrix, IMAGE_SIZE, 0);

    //Create memory buffers
    cl::Buffer VhodnoyMassiv = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(__int32), ImagePixelsMatrix);
    cl::Buffer clmInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector2);
    cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pOutputVector);

    //Build OpenCL program and make the kernel
    
    std::string kernelCode = "";

    std::ifstream fromFile("kernel.cl");
    // Это аналог этого кода - cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    if (fromFile.is_open()) 
    {
        std::string line;
        while (std::getline(fromFile, line)) 
        {
            kernelCode += line + "\n";
        }
        fromFile.close();
        // используйте переменную kernel для дальнейшей обработки исходного кода OpenCL
    }
    else 
    {
        // обработка ошибки открытия файла
    }
    cl::Program program = cl::Program(context, kernelCode);
    cout << "building the kernel... " << endl;
    try
    {
        program.build(contextDevices);
    }
    catch (cl::Error& err) // отлов ошибкиe с выводом исключения
    {
        std::cerr
            << "OpenCL compilation error" << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(contextDevices[0])
            << std::endl;
        throw err;
    }
    cout << "building completed." << endl;

    cl::Kernel kernel(program, "TestKernel");

    //Set arguments to kernel
    int iArg = 0;
    kernel.setArg(iArg++, ImagePixelsMatrix);
    kernel.setArg(iArg++, ResultImagePixelsMatrix);
    //kernel.setArg(iArg++, clmOutputVector);
    kernel.setArg(iArg++, IMAGE_SIZE);

    //Some performance measurement
    //timeValues.clear();
    __int64 start_count;
    __int64 end_count;
    __int64 freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    //Run the kernel on specific ND range
    for (int iTest = 0; iTest < 1; iTest++) // Удалить попытки
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&start_count);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NDRange(128)); // запуск ядра 
        // Попробуй вместо IMAGE_SIZE сделать двухмерный NDRange.
        queue.finish();

        QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
        double time = 1000 * (double)(end_count - start_count) / (double)freq;
        //timeValues.push_back(time);
    }
    //PrintTimeStatistic();
    // Read buffer C into a local list
    queue.enqueueReadBuffer(ImagePixels_device, CL_TRUE, 0, IMAGE_SIZE * sizeof(__int32), ImagePixelsMatrix);
}
