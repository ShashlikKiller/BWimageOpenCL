__kernel void TestKernel
(
  __global const int3* pInputVector, 
  __global double* ResultImagePixelsMatrix, 
  int elementsNumber)
{
    //Get index into global data array
    int iJob = get_global_id(0);

    //Check boundary conditions
    if (iJob >= elementsNumber) return; 

    //Perform calculations
    double result = 0.299 * pInputVector[iJob].z + 0.587 * pInputVector[iJob].y + 0.114 * pInputVector[iJob].x;
    ResultImagePixelsMatrix[iJob] = result;
}