#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

__kernel void TestKernel
(
	__global const int3** ImagePixelsMatrix, 
	__global const int3** ResultImagePixelsMatrix, 
	__global int3** hostImage, 
	int elementsNumber)
{
    //Get index into global data array
    int iJob = get_global_id(0);

    //Check boundary conditions
    if (iJob >= elementsNumber) return; 

    //Perform calculations
    hostImage[iJob]= 0.299 * imageRow[iJob][2] + 0.587 * imageRow[iJob][1] + 0.114 * imageRow[iJob][0];
    //hostImage[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
}