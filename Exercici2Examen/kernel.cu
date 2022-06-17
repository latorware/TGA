
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std; 


__global__ void PrimeraVersioKernel(float* mA, float* mB, float* vC, int n, int m)
{
    int columna = blockIdx.x * blockDim.x + threadIdx.x; 
    if (columna < m)
    {
        for (int i = 0; i < n; i++)
        {
            mA[i * m + columna] = mA[i * m + columna] * vC[i] + mB[i * m + columna] + mA[i * m] * mB[columna];      
        }
    }
}

__global__ void SegonaVersioKernel(float* mA, float* mB, float* vC, int n, int m)
{
    int fila = blockIdx.y * blockDim.y + threadIdx.y; 
    if (fila < n)
    {
        for (int j = 0; j < m; j++)
        {
            mA[fila * m + j] = mA[fila * m + j] * vC[fila] + mB[fila * m + j] + mA[fila * m] * mB[j]; 
        }
    }
}

__global__ void TerceraVersioKernel(float* mA, float* mB, float* vC, int n, int m)
{
    int fila = blockIdx.y * blockDim.y + threadIdx.y; 
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < n && columna < m)
    {
        mA[fila * m + columna] = mA[fila * m + columna] * vC[fila] + mB[fila * m + columna] + mA[fila * m] + mB[columna]; 
    }
}





void PrimeraVersio(float* mA, float* mB, float* vC, int n, int m)
{
    int blocksize = 1024;
    dim3 dimGrid, dimBlock;

    dimBlock.x = blocksize;
    dimBlock.y = 1;
    dimBlock.z = 1;

    dimGrid.x = (m + blocksize - 1) / blocksize;
    dimGrid.y = 1;
    dimGrid.z = 1;

    //Primera columna CPU
    for (int i = 0; i < n; i++)
    {
        mA[i * m] = mA[i * m] * vC[i] + mB[i * m] + mA[i * m] * mB[0];
    }


    float* mADevice;
    float* mBDevice;
    float* vCDevice;
    cudaMalloc((float**)&mADevice, n * m * sizeof(float));
    cudaMalloc((float**)&mBDevice, n * m * sizeof(float));
    cudaMalloc((float**)&vCDevice, n * sizeof(float));

    cudaMemcpy(mADevice, mA, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mBDevice, mB, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vCDevice, vC, n * sizeof(float), cudaMemcpyHostToDevice);

    PrimeraVersioKernel << <dimGrid, dimBlock >> > (mADevice, mBDevice, vCDevice, n, m);

    cudaMemcpy(mA, mADevice, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(mADevice);
    cudaFree(mBDevice);
    cudaFree(vCDevice);

}

void SegonaVersio(float* mA, float* mB, float* vC, int n, int m)
{
    int blocksize = 1024;
    dim3 dimGrid, dimBlock;

    dimBlock.x = 1;
    dimBlock.y = blocksize;
    dimBlock.z = 1;

    dimGrid.x = 1;
    dimGrid.y = (n + blocksize - 1) / blocksize;
    dimGrid.z = 1;


    float* mADevice;
    float* mBDevice;
    float* vCDevice;
    cudaMalloc((float**)&mADevice, n * m * sizeof(float));
    cudaMalloc((float**)&mBDevice, n * m * sizeof(float));
    cudaMalloc((float**)&vCDevice, n * sizeof(float));

    cudaMemcpy(mADevice, mA, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mBDevice, mB, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vCDevice, vC, n * sizeof(float), cudaMemcpyHostToDevice);

    SegonaVersioKernel << <dimGrid, dimBlock >> > (mADevice, mBDevice, vCDevice, n, m);

    cudaMemcpy(mA, mADevice, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(mADevice);
    cudaFree(mBDevice);
    cudaFree(vCDevice);
}

void TerceraVersio(float* mA, float* mB, float* vC, int n, int m)
{
    int blocksize = 32;
    dim3 dimGrid, dimBlock;

    dimBlock.x = blocksize;
    dimBlock.y = blocksize;
    dimBlock.z = 1;

    dimGrid.x = (m + blocksize - 1) / blocksize;
    dimGrid.y = (n + blocksize - 1) / blocksize;
    dimGrid.z = 1;

    //Primera columna CPU
    for (int i = 0; i < n; i++)
    {
        mA[i * m] = mA[i * m] * vC[i] + mB[i * m] + mA[i * m] * mB[0];
    }


    float* mADevice;
    float* mBDevice;
    float* vCDevice;
    cudaMalloc((float**)&mADevice, n * m * sizeof(float));
    cudaMalloc((float**)&mBDevice, n * m * sizeof(float));
    cudaMalloc((float**)&vCDevice, n * sizeof(float));

    cudaMemcpy(mADevice, mA, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mBDevice, mB, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vCDevice, vC, n * sizeof(float), cudaMemcpyHostToDevice);

    TerceraVersioKernel << <dimGrid, dimBlock >> > (mADevice, mBDevice, vCDevice, n, m);

    cudaMemcpy(mA, mADevice, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(mADevice);
    cudaFree(mBDevice);
    cudaFree(vCDevice);
}






int main()
{
    cout << "Tria la versio del kernel" << endl; 

    int numeroKernel; 
    cin >> numeroKernel; 

    int n = 20000; 
    int m = 20000; 

    float* mA = (float*)malloc(n * m * sizeof(float));
    float* mB = (float*)malloc(n * m * sizeof(float));
    float* vC = (float*)malloc(n * sizeof(float)); 

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            mA[i * m + j] = rand() % 100;
            mB[i * m + j] = rand() % 100;
        }
    }
    for (int i = 0; i < n; i++)
    {
        vC[i] = rand() % 100;
    }

    if (numeroKernel == 1)
    {
        PrimeraVersio(mA, mB, vC, n, m); 
    }
    else if (numeroKernel == 2)
    {
        SegonaVersio(mA, mB, vC, n, m);
    }
    else if (numeroKernel == 3)
    {
        TerceraVersio(mA, mB, vC, n, m);
    }

    cout << "Fet" << endl; 
}


