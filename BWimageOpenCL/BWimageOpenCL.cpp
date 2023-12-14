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
#include <malloc.h>

using namespace cv;
using namespace Concurrency;
using namespace std;

float GetGrayPix(int R, int G, int B);
cv::Mat MakeGrayPtrParallel_for(cv::Mat imageOrigin);
cv::Mat MakeGrayByPixel(cv::Mat image);
cv::Mat MakeGrayIGuess(cv::Mat image);
cv::Mat MakeGrayPtr(cv::Mat image);
int** GetImgPixMatrix(cv::Mat image);
cv::Mat GetImgFromPixMatrix(Vec3b** PixMatrix);
Vec3b** GetImgPixChannelMatrix(cv::Mat image);
int*** ConvertToInt3(cv::Vec3b** PixMatrix);
void PerformTestOnDeviceNew(cl::Device device, int*** PixMatrix);

int main(int argc, char* argv[])
{
    const char* filename_example = "C:\\Users\\vipef\\Рабочий стол\\2.png";
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

    Vec3b** PixMatrix = GetImgPixChannelMatrix(image);
    int*** PixMatrixInt = ConvertToInt3(PixMatrix);

#pragma region test
    //for (int i = 0; i < rows; i++)
//{
//    for (int j = 0; j < cols; j++)
//    {
//        for (int z = 0; z < 3; z++)
//        {
//            cout << (PixMatrixInt[i][j][z]) << " ";
//        }
//    }
//}
#pragma endregion

    cv::Mat newImage = GetImgFromPixMatrix(PixMatrix);

    //cv::imshow("cv4", newImage);
    printf("Second");
    image = MakeGrayPtrParallel_for(image);
    //cv::imshow("cv3", image);
    // Сохранение черно-белого изображения
    //cv::imwrite("output_image.jpg", image);
    //waitKey();

    //Get all available platforms
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (unsigned short iPlatform = 0; iPlatform < platforms.size(); iPlatform++)
    {
        //Get all available devices on selected platform
        std::vector<cl::Device> devices;
        platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        cout << iPlatform << endl; // закомментить

        //Perform test on each device
        for (unsigned int iDevice = 0; iDevice < devices.size(); iDevice++)
        {
            try
            {
                PerformTestOnDeviceNew(devices[iDevice], PixMatrixInt);
            }
            catch (cl::Error error)
            {
                std::cout << error.what() << "(" << error.err() << ")" << std::endl;
            }
            //CheckResults();
        }
    }

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

[[Deprecated("This isn't working. Method from docs")]]
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
    cv::Mat _imageResult(_rows, _columns, CV_8UC1);
    for (int x = 0; x < _rows; x++)
    {
        for (int y = 0; y < _columns; y++)
        {
            cv::Vec3b pix = image.at<cv::Vec3b>(x, y);
            float gray = GetGrayPix(pix[0], pix[1], pix[2]); //0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2];
            _imageResult.at<uchar>(x, y) = uchar(gray);
        }
    }
    return _imageResult;
}

cv::Mat GetImgFromPixMatrix(Vec3b** PixMatrix)
{
    int rows = _msize(PixMatrix)/sizeof(Vec3b*);
    int cols = _msize(PixMatrix[0])/sizeof(Vec3b);
    cv::Mat imageResult(rows, cols, CV_8UC3);
    for (int x = 0; x < rows; x++)
    {
        for (int y = 0; y < cols; y++)
        {
            imageResult.at<cv::Vec3b>(x, y) = PixMatrix[x][y];
        }
    }
    return imageResult;
}

Vec3b** GetImgPixChannelMatrix(cv::Mat image)
{
    Vec3b** _pixMat = new Vec3b* [image.rows];
    for (int x = 0; x < image.rows; x++)
    {
        _pixMat[x] = new Vec3b[image.cols];
        for (int y = 0; y < image.cols; y++)
        {
            _pixMat[x][y][0] = image.at<cv::Vec3b>(x, y)[0];
            _pixMat[x][y][1] = image.at<cv::Vec3b>(x, y)[1];
            _pixMat[x][y][2] = image.at<cv::Vec3b>(x, y)[2];
        }
    }    
    return _pixMat;
}

int*** ConvertToInt3(cv::Vec3b** PixMatrix)
{
    int _rows = _msize(PixMatrix) / sizeof(Vec3b*);
    int _cols = _msize(PixMatrix[0]) / sizeof(Vec3b);
    int*** _resultMatrix;
    _resultMatrix = new int** [_rows];

    for (int x = 0; x < _rows; x++) 
    {
        _resultMatrix[x] = new int* [_cols];
        for (int y = 0; y < _cols; y++) 
        {
            _resultMatrix[x][y] = new int[3];
            cv::Vec3b _vec3bVal = PixMatrix[x][y];
            int b = _vec3bVal[0];
            int g = _vec3bVal[1];
            int r = _vec3bVal[2];
            _resultMatrix[x][y][0] = b;
            _resultMatrix[x][y][1] = g;
            _resultMatrix[x][y][2] = r;
        }
    }
    return _resultMatrix;
}

[[Deprecated("Wrong method. Use GetImgPixChannelMatrix instead.")]]
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

int** ResultImagePixelsMatrix;

void PerformTestOnDeviceNew(cl::Device device, int*** PixMatrix)
{
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl << endl;

    const int ROWS = _msize(PixMatrix) / sizeof(int**);
    const int COLS = _msize(PixMatrix[0]) / sizeof(int*);

    const int IMAGE_SIZE = ROWS * COLS;
    //int** pOutputVector; // ON DEVICE (it was int**)
    int*** pInputVector; // ON HOST (it was int**)
    cv::Mat _imgResult(ROWS, COLS, CV_8UC1);

    //For the selected device create a context
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);

    // Создаем очередь для девайса
    cl::CommandQueue queue(context, device);

    pInputVector = PixMatrix;

    //Clean output buffers
    int** pOutputVector = new int* [ROWS];
    for (int i = 0; i < ROWS; i++)
    {
        pOutputVector[i] = new int[COLS];
        for (int j = 0; j < COLS; j++)
        {
            pOutputVector[i][j] = 0;
        }
    }
    //fill_n(pOutputVector, IMAGE_SIZE * sizeof(int), 0);

    //Create memory buffers
    //Sleep(10000);
    cl::Buffer clmInputVector = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * 3 * sizeof(int), pInputVector);
    Sleep(15000);
    cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(int), pOutputVector);

    //Load OpenCL source code
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
    kernel.setArg(iArg++, clmInputVector);
    kernel.setArg(iArg++, clmOutputVector);
    kernel.setArg(iArg++, IMAGE_SIZE);
    //Some performance measurement

    //Run the kernel on specific ND range
    for (int iTest = 0; iTest < 1; iTest++)
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NDRange(128)); // запуск ядра 
        queue.finish();
    }
    // Read buffer C into a local list
    queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, IMAGE_SIZE * sizeof(int), pOutputVector);

}

//
//    //Create memory buffers
//    cl::Buffer VhodnoyMassiv = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(__int32), ImagePixelsMatrix); // ОТПРАВЛЯЕМ ЭТО В КЕРНЕЛ
//    //cl::Buffer clmInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector2);
//    cl::Buffer VihodnoyMassiv = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(__int32), ResultImagePixelsMatrix); // ПОЛУЧАЕМ ЭТО ИЗ МАССИВА
//
//    //Build OpenCL program and make the kernel
//    
//    std::string kernelCode = "";
//
//    std::ifstream fromFile("kernel.cl");
//    // Это аналог этого кода - cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
//    if (fromFile.is_open()) 
//    {
//        std::string line;
//        while (std::getline(fromFile, line)) 
//        {
//            kernelCode += line + "\n";
//        }
//        fromFile.close();
//        // используйте переменную kernel для дальнейшей обработки исходного кода OpenCL
//    }
//    else 
//    {
//        // обработка ошибки открытия файла
//    }
//    cl::Program program = cl::Program(context, kernelCode);
//    cout << "building the kernel... " << endl;
//    try
//    {
//        program.build(contextDevices);
//    }
//    catch (cl::Error& err) // отлов ошибкиe с выводом исключения
//    {
//        std::cerr
//            << "OpenCL compilation error" << std::endl
//            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(contextDevices[0])
//            << std::endl;
//        throw err;
//    }
//    cout << "building completed." << endl;
//
//    cl::Kernel kernel(program, "TestKernel");
//
//    //Set arguments to kernel
//    int iArg = 0;
//    kernel.setArg(iArg++, ImagePixelsMatrix);
//    kernel.setArg(iArg++, ResultImagePixelsMatrix);
//    //kernel.setArg(iArg++, clmOutputVector);
//    kernel.setArg(iArg++, IMAGE_SIZE);
//
//    //Some performance measurement
//    //timeValues.clear();
//    __int64 start_count;
//    __int64 end_count;
//    __int64 freq;
//    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
//
//    //Run the kernel on specific ND range
//    for (int iTest = 0; iTest < 1; iTest++) // Удалить попытки
//    {
//        QueryPerformanceCounter((LARGE_INTEGER*)&start_count);
//
//        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NDRange(128)); // запуск ядра 
//        // Попробуй вместо IMAGE_SIZE сделать двухмерный NDRange.
//        queue.finish();
//
//        QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
//        double time = 1000 * (double)(end_count - start_count) / (double)freq;
//        //timeValues.push_back(time);
//    }
//    //PrintTimeStatistic();
//    // Read buffer C into a local list
//    queue.enqueueReadBuffer(VihodnoyMassiv, CL_TRUE, 0, IMAGE_SIZE * sizeof(__int32), ResultImagePixelsMatrix);
//}
