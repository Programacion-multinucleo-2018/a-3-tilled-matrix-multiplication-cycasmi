//Cynthia Castillo Millán
//A01374530

#include "common.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

using namespace std;
#define N 2000
#define TILE 16

//Initialize data to random numbers
void initRand(float *mat, const float size)
{
    int i;

    srand (time(0));
    for(i = 0; i < size; i++)
    {
        mat[i] = (float)((rand() % 10)+1);
    }

    return;
}

void multMatrixOnHost(float *A, float *B, float *C, const int cols,
                     const int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C[j * rows + i] = 0;
            for (int shared_dim = 0; shared_dim < cols; shared_dim++)
            {
                //dot product
                C[j * rows + i] += A[shared_dim * rows + i] * B[j * rows + shared_dim];
            }
        }
    }

    return;
}

//Tilling on matrix multiplication
__global__ void tilledMatrixMult(float *A, float *B, float *C, int cols, int rows) 
{
    //Indices for the  Matrix
    unsigned int ix = threadIdx.x + blockIdx.x * TILE;
    unsigned int iy = threadIdx.y + blockIdx.y * TILE;

    //Indices for the Tiles
    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;

    //Creating shared memory tiles
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];


    //Initializing shared memory matrices
    for(int i = 0; i < TILE; i ++) 
    {
        for(int j = 0; j < TILE; j++)
        {
            tileA[i][j] = 0;
            tileB[i][j] = 0;
        }
    }

    float cumulativeSum = 0;
    for(int i = (TILE + cols - 1)/TILE; i >= 0; i--) 
    {
        if((i * TILE + x) < cols && (iy < rows))
            tileA[y][x] = A[(iy*rows) + (i*TILE+x)];

        if((i * TILE + y) < rows && (ix < cols))
            tileB[y][x] = B[(i*TILE+y) * cols + ix];

        //Wait for all threads to return their result
        __syncthreads();

        for(int j = 0; j < TILE; j++)
            cumulativeSum += tileA[y][j] * tileB[j][x];

        //Wait for the total result of the threads
        __syncthreads();
    }

    if(ix < cols && iy < rows) 
        C[ix*rows+iy] = cumulativeSum;
}

void checkResult(float *hostRef, float *gpuRef, const int nxy)
{
    double epsilon = 1;
    bool match = 1;

    for (int i = 0; i < nxy; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}


int main(int argc, char **argv)
{
    // set up data size of matrix
    int nx = 0;
    int ny = 0;

    if(argc < 2)
    {
        nx = ny = N;
    }
    else
    {
        nx = ny = stoi(argv[1]);
    }

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side

    initRand(h_A, nxy);
    initRand(h_B, nxy);
    multMatrixOnHost(h_A, h_B, hostRef, nx, ny);


    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnHost elapsed %f ms\n", duration_ms.count());

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    dim3 block(TILE, TILE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    start_cpu =  chrono::high_resolution_clock::now();
    tilledMatrixMult<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();

    duration_ms = end_cpu - start_cpu;

    printf("tilledMatrixMult <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(hostRef, gpuRef, nxy);
    printf("FUNCIONA");

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");


    return (0);
}
