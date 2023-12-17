kernel void imageProcessing(__global int* pInputVector, __global int* ResultImagePixelsMatrix, const int ROWS, const int COLS)
{
    int i = get_global_id(0);

    for (int j = 0; j < 4; j++)
    {
        int index = i * 3 + j;
        if (index + 2 < ROWS * COLS * 3) 
        {
            ResultImagePixelsMatrix[i] = 0.299*pInputVector[index] + 0.587*pInputVector[index+1] + 0.114*pInputVector[index+2];
        }
    }
}