__global__ void relu_kernel(const float* input,
                            float* output,
                            int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // branchless clamp: avoids SIMT warp divergence
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}