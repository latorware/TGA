#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>

#define PINNED 0
#define THREADS 8

void puzzle3DSeq(int Ncar, int Nfil, int Ncol, float *z, float *x, float *y) {
  int i, j, t, ind;

  for (t=0; t<Ncar; t++)
    for (i=0; i<Nfil; i++)
      for (j=0; j<Ncol; j++) {
        ind = t*Nfil*Ncol + i*Ncol + j;
        z[ind] = 0.5*x[ind] + 0.75*y[ind] + x[ind]*y[ind];
      }
}

__global__ void puzzle3DPAR1x1x1(int Ncar, int Nfil, int Ncol, float *z, float *x, float *y) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int t = blockIdx.z * blockDim.z + threadIdx.z;
  int ind = t*Nfil*Ncol + i*Ncol + j;

  if (t<Ncar && i<Nfil && j<Ncol)
    z[ind] = 0.5*x[ind] + 0.75*y[ind] + x[ind]*y[ind];

  // Aqui va vuestro codigo
}


void InitM(int Ncar, int Nfil, int Ncol, float *v);
int Test3D(int Ncar, int Nfil, int Ncol, float *zseq, float *zpar);
void CheckCudaError(char sms[], int line);
float GetTime(void);        



int main(int argc, char** argv) {
  unsigned int Ncar, Nfil, Ncol;
  unsigned int numBytes;
 
  cudaEvent_t E1, E2, E3, E4, E5, E6;
  float TiempoK, TiempoSEQ;
  float t1,t2;

  float *hX, *hY, *hZ, *mZ;
  float *dX, *dY, *dZ;

  int gpu, count;
  char test;

  // Dimension de la matriz resultado y comprobacion resultado
  if (argc == 1)      { test = 'N';      Ncar = 128;           Nfil = 64;           Ncol = 128;}
  else if (argc == 4) { test = 'N';      Ncar = atoi(argv[1]); Nfil = atoi(argv[2]); Ncol = atoi(argv[3]); }
  else if (argc == 5) { test = *argv[4]; Ncar = atoi(argv[1]); Nfil = atoi(argv[2]); Ncol = atoi(argv[3]); }
  else { printf("Usage: ./exe Ncar Ncol Nfil test\n"); exit(0); }

  // Esta porcion de codigo la explicaremos en clase
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  gpu = rand();
  cudaSetDevice((gpu>>3) % count);

  numBytes = Ncar * Ncol * Nfil * sizeof(float);

  cudaEventCreate(&E1); cudaEventCreate(&E2);
  cudaEventCreate(&E3); cudaEventCreate(&E4);
  cudaEventCreate(&E5); cudaEventCreate(&E6);

  if (PINNED) {
    // Obtiene Memoria [pinned] en el host
    cudaMallocHost((float**)&hX, numBytes); 
    cudaMallocHost((float**)&hY, numBytes); 
    cudaMallocHost((float**)&hZ, numBytes);
    cudaMallocHost((float**)&mZ, numBytes);
  }
  else {
    // Obtener Memoria en el host
    hX = (float*) malloc(numBytes); 
    hY = (float*) malloc(numBytes); 
    hZ = (float*) malloc(numBytes);  
    mZ = (float*) malloc(numBytes);  
  }

  // Inicializa las matrices
  InitM(Ncar, Nfil, Ncol, hX);
  InitM(Ncar, Nfil, Ncol, hY);

  // Ejecucion Secuencial, se ejecuta varias veces para evitar problemas de precision con el clock
  t1=GetTime();
  for (int t = 0; t<20; t++)
    puzzle3DSeq(Ncar, Nfil, Ncol, mZ, hX, hY);
  t2=GetTime();
  TiempoSEQ = (t2 - t1) / 20.0;
 
  // Obtener Memoria en el device
  cudaMalloc((float**)&dX, numBytes); 
  cudaMalloc((float**)&dY, numBytes); 
  cudaMalloc((float**)&dZ, numBytes); 
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(dX, hX, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dY, hY, numBytes, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  //
  // KERNEL ELEMENTO a ELEMENTO
  //

  int nThreads = THREADS;
  int nBlocksFil = (Nfil+nThreads-1)/(nThreads);
  int nBlocksCol = (Ncol+nThreads-1)/nThreads;
  int nBlocksCar = (Ncar+nThreads-1)/nThreads;

  //dim3 dimGridE(Ncol/nThreads, Nfil/nThreads, 1);
  dim3 dimGridE(nBlocksCol, nBlocksFil, nBlocksCar);
  dim3 dimBlockE(nThreads, nThreads, nThreads);

  printf("\n");
  printf("Kernel Elemento a Elemento\n");
  printf("%d caras x %d filas x %d columnas\n", Ncar, Nfil, Ncol);
  printf("%d elementos\n", Ncar*Nfil*Ncol);
  printf("%d x %d x %d threads\n", nThreads, nThreads, nThreads);
  printf("%d x %d x %d blocks\n", nBlocksCol, nBlocksFil, nBlocksCar);

  cudaEventRecord(E5, 0);
  cudaEventSynchronize(E5);

  // Ejecutar el kernel elemento a elemento
  puzzle3DPAR1x1x1<<<dimGridE, dimBlockE>>>(Ncar, Nfil, Ncol, dZ, dX, dY);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E6, 0);
  cudaEventSynchronize(E6);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en vZe para poder comprobar el resultado
  cudaMemcpy(hZ, dZ, numBytes, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Comprobar resultado
  if (test == 'Y') { if (Test3D(Ncar, Nfil, Ncol, hZ, mZ)) printf ("ELEMENTOS - TEST PASS\n"); else printf ("ELEMENTOS - TEST FAIL\n"); }


  // Liberar Memoria del device 
  cudaFree(dX); cudaFree(dY); cudaFree(dZ);

  cudaDeviceSynchronize();


  cudaEventElapsedTime(&TiempoK, E5, E6);
 
  cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaEventDestroy(E4); cudaEventDestroy(E5); cudaEventDestroy(E6);

  printf("\n");
  printf("Resumen Rendimiento\n");
  printf("Tiempo Paralelo Kernel elemento a elemento: %4.6f ms (%4.2f GFLOPS)\n", TiempoK, ((float) 5*Ncar*Nfil*Ncol) / (1000000.0 * TiempoK));
  printf("Tiempo Secuencial: %4.6f milseg (%4.2f GFLOPS)\n", TiempoSEQ, ((float) 5*Ncar*Nfil*Ncol) / (1000000.0 * TiempoSEQ));
  printf("\n\n\n");


  if (PINNED) { cudaFreeHost(hZ); cudaFreeHost(mZ); cudaFreeHost(hX); cudaFreeHost(hY); }
  else { free(hZ); free(mZ); free(hX); free(hY); }

}


void InitM(int Ncar, int Nfil, int Ncol, float *M) {
   int i;
   for (i=0; i<Ncar*Nfil*Ncol; i++) 
     M[i] = rand();
   
}
int error(float a, float b) {

  if (abs (a - b) / a > 0.000001) return 1;
  else  return 0;

}

int Test3D(int Ncar, int Nfil, int Ncol, float *vZ, float *hZ) {
   int t, i, j, ind;
   ind = 0;
   for (t=0; t<Ncar; t++) 
     for (i=0; i<Nfil; i++) 
       for (j=0; j<Ncol; j++) {
      
         if (error(vZ[ind], hZ[ind])) {
           printf ("%d-%d: %f - %f = %f \n", i, j, vZ[ind], hZ[ind], vZ[ind] - hZ[ind]);
           return 0;
         }
         ind++;
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
  //else printf("(OK) %s \n", sms);
}

float GetTime(void)        {
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}

