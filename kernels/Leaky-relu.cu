__global__ void leaky_relu(const float* input, float* output, int N, float alpha){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < N){
        float x = input[idx]; 
        output[idx] = (x > 0.0f) ? x : alpha * x; 
    }
}