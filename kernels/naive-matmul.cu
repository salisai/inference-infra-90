// M, N are the input matrices and P is the output
__global__ void MatrixMulKernel(const float* M, const float* N, float* P, int Width){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < Width && col < Width){ //boundary check for square matrix
        float Pvalue = 0.0f; 
        for (int k = 0; k < Width; ++k){
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }

        P[row * Width + col] = Pvalue; //one write
    }
}