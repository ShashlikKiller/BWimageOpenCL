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
cv::Mat GetGrayImg(int** PixMatrix);
int* ResultImagePixelsMatrix;
int** convertTo2D(int* resultImageMatrix, int width, int height);

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
    //cv::Mat newImage = GetImgFromPixMatrix(PixMatrix);
    //cv::imshow("cv3", newImage);
    int rows = image.size[0];
    int cols = image.size[1];

#pragma region test
   /* for (int i = 0; i < rows; i++)
{
    for (int j = 0; j < cols; j++)
    {
        for (int z = 0; z<3; z++)
            cout << (PixMatrixInt[i][j][z]) << " ";
    }
}*/
#pragma endregion


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
    int** GrayPixMatrix = convertTo2D(ResultImagePixelsMatrix, cols, rows);
    cout << _msize(GrayPixMatrix) / sizeof(int*) << endl;
    //cout << _msize(ResultImagePixelsMatrix[49]) / sizeof(int) << endl;
    //for (int i = 1; i < rows-1; i++)
    //{
    //    for (int j = 1; j < cols-1; j++)
    //    {
    //        cout << GrayPixMatrix[i][j] << " ";
    //    }
    //}
    cv::Mat image_cv = MakeGrayPtrParallel_for(image);
    cv::Mat image_cl = GetGrayImg(GrayPixMatrix);
    cv::imshow("cv", image_cv);
    cv::imshow("cl",image_cl);
    waitKey();
    // освобождаем ресурсы
    image.cv::Mat::release();
    // удаляем окно
    cv::destroyWindow("cv");
}

int* convertTo1D(int rows, int cols, int*** pixMatrix)
{
    int* arr_1d = new int[rows * cols * 3];
    int index = 0;
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            for (int k = 0; k < 3; k++) 
            {
                arr_1d[index] = pixMatrix[i][j][k];
                index++;
            }
        }
    }
    return arr_1d;
}

int** convertTo2D(int* resultImageMatrix, int width, int height) 
{
    if (_msize(resultImageMatrix) / sizeof(int) != width * height) 
    {
        throw std::invalid_argument("The size of the one-dimensional array does not match the specified width and height");
    }

    int** outputImageMatrix = new int* [height];
    for (int i = 0; i < height; i++) {
        outputImageMatrix[i] = new int[width];
    }

    int index = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            outputImageMatrix[i][j] = resultImageMatrix[index];
            index++;
        }
    }

    return outputImageMatrix;
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
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl << endl;

    const int ROWS = _msize(PixMatrix) / sizeof(int**);
    const int COLS = _msize(PixMatrix[0]) / sizeof(int*);

    const int IMAGE_SIZE = ROWS * COLS;
    int* pInputVector; // ON HOST (it was int**)

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
    cl::Kernel kernel(program, "imageProcessing");

    //Set arguments to kernel
    int iArg = 0;
    kernel.setArg(iArg++, clmInputVector);
    kernel.setArg(iArg++, clmOutputVector);
    //kernel.setArg(iArg++, IMAGE_SIZE);
    kernel.setArg(iArg++, ROWS);
    kernel.setArg(iArg++, COLS);
    //Some performance measurement

    //Run the kernel on specific ND range
    for (int iTest = 0; iTest < 1; iTest++)
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NDRange(50)); // запуск ядра 
        queue.finish();
    }
    // Read buffer C into a local list
    queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, IMAGE_SIZE * sizeof(int), ResultImagePixelsMatrix);
}
