__global__ void matrixAddition(const float* A, const float* B, float* C, int M, int N){
    int i = blockIdx.y * blockDim.y + threadIdx.y; // along rows
    int j = blockIdx.x * blockDim.x + threadIdx.x; // along cols

    if(i < M && j < M){
        int idx = i * N + j; 
        C[idx] = A[idx] + B[Idx];
    }
}