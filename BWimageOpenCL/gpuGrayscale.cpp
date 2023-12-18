#include "gpuGrayscale.h"

int* GrayscaledImage;
double start, elapsed;

void grayscaleOnKernel(int*** pixMatrix)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (unsigned short iPlatform = 0; iPlatform < platforms.size(); iPlatform++)
    {
        //Get all available devices on selected platform
        std::vector<cl::Device> devices;
        platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (unsigned int iDevice = 0; iDevice < devices.size(); iDevice++)
        {
            try
            {
                return grayscaleKernelExecution(devices[iDevice], pixMatrix);
            }
            catch (cl::Error error)
            {
                std::cout << error.what() << "(" << error.err() << ")\n";
            }
        }
    }
}

void grayscaleKernelExecution(cl::Device device, int*** PixMatrix)
{
    std::cout << "Device for OpenCL calculations: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    const int ROWS = _msize(PixMatrix) / sizeof(int**);
    const int COLS = _msize(PixMatrix[0]) / sizeof(int*);

    const int IMAGE_SIZE = ROWS * COLS;
    int* imageChannelMatrix;

    //For the selected device create a context
    std::vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);

    // Создаем очередь для девайса
    cl::CommandQueue queue(context, device);

    imageChannelMatrix = convertTo1D(ROWS, COLS, PixMatrix);

    //Clean output buffers
    GrayscaledImage = new int[IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++)
    {
        GrayscaledImage[i] = 1;
    }
    //fill_n(pOutputVector, IMAGE_SIZE * sizeof(int), 0);

    //Create memory buffers
    cl::Buffer clmInputVector;
    cl::Buffer clmOutputVector;
    Sleep(1000);
    clmInputVector = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * 3 * sizeof(int), imageChannelMatrix);
    Sleep(1000);
    clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, IMAGE_SIZE * sizeof(int), GrayscaledImage);

    //Load OpenCL source code
    std::string kernelCode = "";
    std::ifstream fromFile("kernel.cl");  // Это аналог этого кода - cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
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
        std::cout << "Error: can't open file with kernel's code.\n";
    }
    cl::Program program = cl::Program(context, kernelCode);
    std::cout << "building the kernel... \n";
    try
    {
        program.build(contextDevices);
    }
    catch (cl::Error& err) // отлов ошибкиe с выводом исключения
    {
        std::cerr
            << "OpenCL compilation error\n"
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(contextDevices[0])
            << std::endl;
        throw err;
    }
    std::cout << "building completed." << std::endl;
    cl::Kernel kernel(program, "imageProcessing");
    //Set arguments to kernel
    start = (double)getTickCount();
    int iArg = 0;
    kernel.setArg(iArg++, clmInputVector);
    kernel.setArg(iArg++, clmOutputVector);
    //kernel.setArg(iArg++, IMAGE_SIZE);
    kernel.setArg(iArg++, ROWS);
    kernel.setArg(iArg++, COLS);
    //Run the kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NDRange(50)); // запуск ядра 
    queue.finish();
    // Buffer going from kernel to global memory
    queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, IMAGE_SIZE * sizeof(int), GrayscaledImage);
    elapsed = ((double)getTickCount() - start) / getTickFrequency();
}

int* getGrayscaledImageFromDevice()
{
    return GrayscaledImage;
}

double getDeviceElapsed()
{
    return elapsed;
}