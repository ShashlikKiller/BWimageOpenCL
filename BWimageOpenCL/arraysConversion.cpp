#include "arraysConversion.h"

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

int*** convertTo3D(cv::Vec3b** PixMatrix)
{
    int _rows = _msize(PixMatrix) / sizeof(cv::Vec3b*);
    int _cols = _msize(PixMatrix[0]) / sizeof(cv::Vec3b);
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

cv::Mat GetGrayscaledImage(int** PixMatrix)
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

cv::Vec3b** GetImgPixChannelMatrix(cv::Mat image)
{
    cv::Vec3b** _pixMat = new cv::Vec3b * [image.rows];
    for (int x = 0; x < image.rows; x++)
    {
        _pixMat[x] = new cv::Vec3b[image.cols];
        for (int y = 0; y < image.cols; y++)
        {
            _pixMat[x][y][0] = image.at<cv::Vec3b>(x, y)[0];
            _pixMat[x][y][1] = image.at<cv::Vec3b>(x, y)[1];
            _pixMat[x][y][2] = image.at<cv::Vec3b>(x, y)[2];
        }
    }
    return _pixMat;
}
