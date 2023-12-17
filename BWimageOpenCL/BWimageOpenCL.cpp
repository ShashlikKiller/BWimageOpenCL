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
cv::Mat GetGrayImg(int** PixMatrix);
int* ResultImagePixelsMatrix;
int** convertTo2D(int* resultImageMatrix, int width, int height);
double time_start, total_time;

int ROWS, COLS;

int main(int argc, char* argv[])
{
    std::string path = __FILE__;
    size_t pos = path.find_last_of('/');
    // Выделяем подстроку с относительным путем к файлу
    std::string filename_example = path.substr(0, pos + 1) + "example.jpg";

    // Получаем относительный путь к файлу
    std::string filename = (argc == 2) ? argv[1] : filename_example;
    // Загрузка изображения
    cv::Mat imageOrigin = cv::imread(filename);
    // Проверка, что изображение загружено успешно
    if (imageOrigin.empty()) 
    {
        std::cout << "Can't load this image." << std::endl;
        return -1;
    }
    cv::Mat image = imageOrigin.cv::Mat::clone();
    cv::Mat image_cvtColor;


    // Далее идет работа с клоном изображения, чтобы с оригинальным ничего не произошло
    // даже если мы явно присвоим изображения через image A = image B, то А будет просто ссылкой на В и при изменении В измениться и А.

    Vec3b** PixMatrix = GetImgPixChannelMatrix(image);
    int*** PixMatrixInt = ConvertToInt3(PixMatrix);
    ROWS = image.size[0];
    COLS = image.size[1];

    //Get all available platforms
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (unsigned short iPlatform = 0; iPlatform < platforms.size(); iPlatform++)
    {
        //Get all available devices on selected platform
        std::vector<cl::Device> devices;
        platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
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
        }
    }
    int** GrayPixMatrix = convertTo2D(ResultImagePixelsMatrix, ROWS, COLS);

    //Making an image from CL matrix:
    time_start = (double)getTickCount();
    cv::Mat image_cl = GetGrayImg(GrayPixMatrix);
    total_time = ((double)getTickCount() - time_start) / getTickFrequency();
    std::cout << "Building the image from CL matrix: " << total_time << " seconds." << std::endl;

    //MakeGrayPtrParallel_for
    time_start = (double)getTickCount();
    cv::Mat image_makeGrayPtrParallel_for = MakeGrayPtrParallel_for(image);
    total_time = ((double)getTickCount() - time_start) / getTickFrequency();
    std::cout << "Time for MakeGrayPtrParallel_for: " << total_time << " seconds." << std::endl;

    //MakeGrayPtr
    time_start = (double)getTickCount();
    cv::Mat image_makeGrayPtr = MakeGrayPtr(image);
    total_time = ((double)getTickCount() - time_start) / getTickFrequency();
    std::cout << "Time for MakeGrayPtr: " << total_time << " seconds." << std::endl;

    //MakeGrayByPixel
    time_start = (double)getTickCount();
    cv::Mat image_makeGrayByPixel = MakeGrayByPixel(image);
    total_time = ((double)getTickCount() - time_start) / getTickFrequency();
    std::cout << "Time for MakeGrayByPixel: " << total_time << " seconds." << std::endl;

    //cv::cvtColor
    time_start = (double)getTickCount();
    cv::cvtColor(image, image_cvtColor, cv::COLOR_BGR2GRAY);
    total_time = ((double)getTickCount() - time_start) / getTickFrequency();
    std::cout << "\n\n\nTime for cvtColor: " << total_time << " seconds." << std::endl;

    waitKey();

    // Показ готовых изображений:
    //cv::imshow("cv_cvtColor", image);
    //cv::imshow("cv_parallelFor", image_cv);
    //cv::imshow("cl_kernel",image_cl);

    // освобождаем ресурсы
    image.cv::Mat::release();
    imageOrigin.cv::Mat::release();
    image_makeGrayByPixel.cv::Mat::release();
    image_cvtColor.cv::Mat::release();
    image_makeGrayPtr.cv::Mat::release();
    image_cl.cv::Mat::release();
    image_makeGrayPtrParallel_for.cv::Mat::release();
    
    // Удаление окон с изображениями:
    //cv::destroyWindow("cv_cvtColor");
    //cv::destroyWindow("cv_parallelFor");
    //cv::destroyWindow("cl_kernel");
}

int* convertTo1D(int rows, int cols, int*** pixMatrix)
{
    int* _result1d = new int[rows * cols * 3];
    int _resultIndex = 0;
    for (int x = 0; x < rows; x++) 
    {
        for (int y = 0; y < cols; y++) 
        {
            for (int z = 0; z < 3; z++) 
            {
                _result1d[_resultIndex] = pixMatrix[x][y][z];
                _resultIndex++;
            }
        }
    }
    return _result1d;
}

int** convertTo2D(int* resultImageMatrix, int width, int height) 
{
    if (_msize(resultImageMatrix) / sizeof(int) != (width * height)) 
    {
        throw std::invalid_argument("The size of the one-dimensional array does not match the specified width and height");
    }

    int** _outputImageMatrix = new int* [height];
    for (int i = 0; i < height; i++) 
    {
        _outputImageMatrix[i] = new int[width];
    }
    int index = 0;
    for (int i = 0; i < height; i++) 
    {
        for (int j = 0; j < width; j++) 
        {
            _outputImageMatrix[i][j] = resultImageMatrix[index];
            index++;
        }
    }
    return _outputImageMatrix;
}

cv::Mat MakeGrayPtr(cv::Mat image)
{
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

cv::Mat MakeGrayPtrParallel_for(cv::Mat image)
{
    cv::Mat imageGray(image.rows, image.cols, CV_8UC1);
    parallel_for(size_t(0), size_t(image.rows), [&image, &imageGray](size_t x) // добавляем image в список захвата
        {
            uchar _imgPix;
            Vec3b* _imageOriginRow = image.ptr<Vec3b>(x);
            uchar* _imgRow = imageGray.ptr<uchar>(x);
            for (int y = 0; y < image.cols; y++)
            {
                _imgPix = GetGrayPix(_imageOriginRow[y][2], _imageOriginRow[y][1], _imageOriginRow[y][0]);
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
            float gray = GetGrayPix(pix[0], pix[1], pix[2]);
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
            _resultMatrix[x][y][0] = _vec3bVal[0];
            _resultMatrix[x][y][1] = _vec3bVal[1];
            _resultMatrix[x][y][2] = _vec3bVal[2];
        }
    }
    return _resultMatrix;
}

cv::Mat GetGrayImg(int** PixMatrix)
{
    int _rows = _msize(PixMatrix) / sizeof(int*);
    int _cols = _msize(PixMatrix[0]) / sizeof(int);
    cv::Mat _image(_rows, _cols, CV_8UC1);
    for (int x = 0; x < _rows; x++)
    {
        for (int y = 0; y < _cols; y++)
        {
            _image.at<uchar>(x, y) = PixMatrix[x][y];
        }
    }
    return _image;
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

void PerformTestOnDeviceNew(cl::Device device, int*** PixMatrix)
{
    std::cout << "GPU for OpenCL calculations: " << device.getInfo<CL_DEVICE_NAME>() << endl;

    //const int ROWS = _msize(PixMatrix) / sizeof(int**);
    //const int COLS = _msize(PixMatrix[0]) / sizeof(int*);

    const int IMAGE_SIZE = ROWS * COLS;
    int* pInputVector; // ON HOST

    //For the selected device create a context
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);

    // Создаем очередь для девайса
    cl::CommandQueue queue(context, device);

    pInputVector = convertTo1D(ROWS, COLS, PixMatrix);

    //Clean output buffers
    ResultImagePixelsMatrix = new int[IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++)
    {
        ResultImagePixelsMatrix[i] = 1;
    }
    //fill_n(pOutputVector, IMAGE_SIZE * sizeof(int), 0);

    //Create memory buffers
    cl::Buffer clmInputVector;
    cl::Buffer clmOutputVector;
    Sleep(1000);
    clmInputVector = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * 3 * sizeof(int), pInputVector);
    Sleep(1000);
    clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(int), ResultImagePixelsMatrix);


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
    }
    else
    {
        std::printf("Error: file with kernel can't be opened.\n");
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
    std::cout << "building completed." << endl;
    cl::Kernel kernel(program, "imageProcessing");

    time_start = (double)getTickCount();
    //Set arguments to kernel
    int iArg = 0;
    kernel.setArg(iArg++, clmInputVector);
    kernel.setArg(iArg++, clmOutputVector);
    //kernel.setArg(iArg++, IMAGE_SIZE);
    kernel.setArg(iArg++, ROWS);
    kernel.setArg(iArg++, COLS);

    //Run the kernel on specific ND range
    for (int iTest = 0; iTest < 1; iTest++)
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NDRange(50)); // запуск ядра 50()
        queue.finish();
    }
    // Read buffer C into a local list
    queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, IMAGE_SIZE * sizeof(int), ResultImagePixelsMatrix);

    total_time = ((double)getTickCount() - time_start) / getTickFrequency();
    std::cout << "Time for kernel imageProcessing: " << total_time << " seconds." << std::endl;
}
