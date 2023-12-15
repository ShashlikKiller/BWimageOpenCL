kernel void imageProcessing(__global int* pInputVector, __global int* ResultImagePixelsMatrix, const int ROWS, const int COLS)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < ROWS && j < COLS) {
        int index = j * ROWS * 3 + i * 3;
        int product = 1;
        for (int k = 0; k < 3; k++) 
        {
            product *= pInputVector[index + k];
        }
        ResultImagePixelsMatrix[j * ROWS + i] = product;
    }
    //int iJob = get_global_id(0);
    //int iJob2 = get_global_id(1);
    //int iJob3 = get_global_id(2);
    //int indexInput = iJob * COLS * DEPTH + iJob2 * DEPTH + iJob3;
    
    //int result = (int)(0.299 * pInputVector[index * 3] + 0.587 * pInputVector[index * 3 + 1] + 0.114 * pInputVector[index * 3 + 2]);
    //ResultImagePixelsMatrix[index] = result;
    //printf(ResultImagePixelsMatrix[index]);
}

//__kernel void TestKernel
//(
//  __global const int*** pInputVector, 
//  __global int ResultImagePixelsMatrix[][], 
//  int elementsNumber)
//{
//    //Get index into global data array
//    int iJob = get_global_id(0);
 //   int iJob2 = get_global_id(1);

    //Check boundary conditions
    //if (iJob >= elementsNumber) return; 

    //Perform calculations
    //int result = 0.299 * pInputVector[iJob][iJob2][0] + 0.587 * pInputVector[iJob][iJob2][1] + 0.114 * pInputVector[iJob][iJob2][2];
    //ResultImagePixelsMatrix[iJob][iJob2] = result;
//}