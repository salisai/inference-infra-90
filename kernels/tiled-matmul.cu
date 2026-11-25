#define TILE_WIDTH 2

__global__ void matMulKernel(float* M, float* N, float* P, int Width){ 
    //Width: of the matrix, here 4

    // one tile-worth of M and one tile-worth of N, 
    // shared by every thread in the block(scope). 
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; //block column position
    int by = blockIdx.y; //block row position, 

    //every thread in the same block has the same bx and by. 

    int tx = threadIdx.x; //column position of thread within the block. 
    int ty = threadIdx.y; //row position within the block. 

    int Row = by * TILE_WIDTH + ty; // global row index of the thread
    int Col = bx * TILE_WIDTH + tx; // global column index of the thread

    float Pvalue = 0; 
    //one iteration per tile. 
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph){
        
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        //Mds[1][0] = M[1*4 + 0*2 + 0] = M[4] = M1,0

        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        //Nds[0][1] = N[(0*2 + 0)*4 + 1] = N[1] = N0,1

        __syncthreads();
        //read-after-write dependence(true dependence): threads must wait until 
        // for data to be written to the proper place by other threads 
        // before they try to read it. 

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
        // write-after-read dependence(false dependence): thread 
        // must wait for the data to be read by all threads that 
        // need it before overwriting it. 
    }

    //global, the full complete value is writen here, not the half dot product value. 
    P[Row*Width + Col] = Pvalue; 
}