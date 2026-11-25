__global__ void tanh_kernel(const float* input, float* output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if(idx < N){
        //using the hardware intrinsic tanhf() for speed and stability
        output[idx] = tanhf(input[idx])
    }
}