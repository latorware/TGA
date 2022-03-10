#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>

#define PINNED 0
#define THREADS 1024

void puzzle1DSeq(int N, float *z, float *x, float *y) {
  int i;
  for (i=0; i<N; i++)
    z[i] = 0.5*x[i] + 0.75*y[i] + x[i]*y[i];
}

__global__ void puzzle1DPAR(int N, float *z, float *x, float *y) {

   // Aqui va vuestro codigo

}


void InitV(int N, float *v);
int Test1D(int N, float *zseq, float *zpar);
void CheckCudaError(char sms[], int line);
float GetTime(void);        



int main(int argc, char** argv)

{
  unsigned int N;
  unsigned int numBytes;
  unsigned int nBlocks, nThreads;
  int count, gpu;

  cudaEvent_t E0, E1, E2, E3;
  float TiempoTotal, TiempoKernel;
  float t1,t2;

  float *hX, *hY, *hZ, *vZ;

  float *dX, *dY, *dZ;

  char test;

  // Dimension del vector y comprobacion resultado
  if (argc == 1)      { test = 'N'; N = 8 * 2048; }
  else if (argc == 2) { test = 'N'; N = atoi(argv[1]); }
  else if (argc == 3) { test = *argv[2]; N = atoi(argv[1]); }
  else { printf("Usage: ./exe TAM test\n"); exit(0); }

  // Esta porcion de codigo la explicaremos en clase
  cudaGetDeviceCount(&count);
  srand(time(NULL)); 
  gpu = rand(); 
  cudaSetDevice((gpu>>3) % count);


  N = N * 1024;
  //N = N - 1;    //Descomentar esta linea para probar dimensiones NO multiplo del #threads

  nThreads = THREADS;
  nBlocks = N/nThreads;  // Solo funciona bien si N multiplo de nThreads
  numBytes = N * sizeof(float);

  dim3 dimGrid(nBlocks, 1, 1);
  dim3 dimBlock(nThreads, 1, 1);

  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  if (PINNED) {
    // Obtiene Memoria [pinned] en el host
    cudaMallocHost((float**)&hX, numBytes); 
    cudaMallocHost((float**)&hY, numBytes); 
    cudaMallocHost((float**)&hZ, numBytes);
    cudaMallocHost((float**)&vZ, numBytes);
  }
  else {
    // Obtener Memoria en el host
    hX = (float*) malloc(numBytes); 
    hY = (float*) malloc(numBytes); 
    hZ = (float*) malloc(numBytes);  
    vZ = (float*) malloc(numBytes);  
  }

  // Inicializa los vectores
  InitV(N, hX);
  InitV(N, hY);

 
  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);
 
  // Obtener Memoria en el device
  cudaMalloc((float**)&dX, numBytes); 
  cudaMalloc((float**)&dY, numBytes); 
  cudaMalloc((float**)&dZ, numBytes); 
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(dX, hX, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dY, hY, numBytes, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel 
  puzzle1DPAR<<<dimGrid, dimBlock>>>(N, dZ, dX, dY);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en vZ para poder comprobar el resultado
  cudaMemcpy(vZ, dZ, numBytes, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del device 
  cudaFree(dX); cudaFree(dY); cudaFree(dZ);

  cudaDeviceSynchronize();

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
 
  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  t1=GetTime();
  puzzle1DSeq(N, hZ, hX, hY);
  t2=GetTime();


  printf("N Elementos: %d\n", N);
  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);
  printf("Tiempo Paralelo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Paralelo Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Tiempo Secuencial: %4.6f milseg\n", t2-t1);
  printf("Rendimiento Paralelo Global: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * TiempoTotal));
  printf("Rendimiento Paralelo Kernel: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * TiempoKernel));
  printf("Rendimiento Secuencial: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * (t2 - t1)));


  if (test == 'N')
    printf ("NO TEST\n");
  else if (Test1D(N, hZ, vZ))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");
  printf("------------------------------------\n");

  if (PINNED) {
    cudaFreeHost(hZ); cudaFreeHost(vZ); cudaFreeHost(hX); cudaFreeHost(hY); 
  }
  else {
    free(hZ); free(vZ); free(hX); free(hY);
  }


}


void InitV(int N, float *v) {
   int i;
   for (i=0; i<N; i++) 
     v[i] = rand();
   
}
int error(float a, float b) {

  if (abs (a - b) / a > 0.000001) return 1;
  else  return 0;

}

int Test1D(int N, float *vZ, float *hZ) {
   int i;

   for (i=0; i<N-THREADS; i = i+423) {
     if (error(vZ[i], hZ[i])) {
       printf ("%d: %f - %f = %f \n", i, vZ[i], hZ[i], vZ[i] - hZ[i]);
       return 0;
     }
   }

   // Comprobacion de las ultimas posiciones
   for (i=N-THREADS; i<N; i++) {
     if (error(vZ[i], hZ[i])) {
       printf ("%d: %f - %f = %f \n", i, vZ[i], hZ[i], vZ[i] - hZ[i]);
       return 0;
     }
   }
   return 1;
}

void CheckCudaError(char sms[], int line) {
  cudaError_t error;
 
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
}

float GetTime(void)        {
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}


