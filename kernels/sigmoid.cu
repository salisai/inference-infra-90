__global__ void sigmoid_kernel(const float* input,
                               float* output,
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (idx < N) {
        float x = input[idx];
        // expf() evaluates e^(-x) using hardware Special Function Units (SFUs)
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}