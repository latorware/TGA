#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>

#define PINNED 0
#define THREADS 16

void puzzle2DSeq(int Nfil, int Ncol, float *z, float *x, float *y) {
  int i, j, ind;
  for (i=0; i<Nfil; i++)
    for (j=0; j<Ncol; j++) {
      ind = i * Ncol + j;
      z[ind] = 0.5*x[ind] + 0.75*y[ind] + x[ind]*y[ind];
    }
}

__global__ void puzzle2DPAR1x1(int Nfil, int Ncol, float *z, float *x, float *y) {

  // Aqui va vuestro codigo
}

__global__ void puzzle2DPARfil(int Nfil, int Ncol, float *z, float *x, float *y) {

  // Aqui va vuestro codigo
}

__global__ void puzzle2DPARcol(int Nfil, int Ncol, float *z, float *x, float *y) {

  // Aqui va vuestro codigo
}


void InitM(int Nfil, int Ncol, float *v);
int Test2D(int Nfil, int Ncol, float *zseq, float *zpar);
void CheckCudaError(char sms[], int line);
float GetTime(void);        



int main(int argc, char** argv) {
  unsigned int Nfil, Ncol;
  unsigned int numBytes;
  unsigned int nBlocks, nThreads;
 
  cudaEvent_t E1, E2, E3, E4, E5, E6;
  float TiempoKfil, TiempoKcol, TiempoKele, TiempoSEQ;
  float t1,t2;

  float *hX, *hY, *hZ, *mZ;

  float *dX, *dY, *dZ;

  int gpu, count;
  char test;

  // Dimension de la matriz resultado y comprobacion resultado
  if (argc == 1)      { test = 'N';      Nfil = 512;           Ncol = 768;}
  else if (argc == 3) { test = 'N';      Nfil = atoi(argv[1]); Ncol = atoi(argv[2]); }
  else if (argc == 4) { test = *argv[3]; Nfil = atoi(argv[1]); Ncol = atoi(argv[2]); }
  else { printf("Usage: ./exe Ncol Nfil test\n"); exit(0); }


  // Esta porcion de codigo la explicaremos en clase
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  gpu = rand();
  cudaSetDevice((gpu>>3) % count);


  // Descomentar para dimensiones NO multiplo del numero de threads
  // if (Ncol % THREADS == 0) Ncol--; 
  // if (Nfil % THREADS == 0) Nfil--; 

  numBytes = Ncol * Nfil * sizeof(float);

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
  InitM(Nfil, Ncol, hX);
  InitM(Nfil, Ncol, hY);

  // Ejecucion Secuencial, se ejecuta varias veces para evitar problemas de precision con el clock
  t1=GetTime();
  for (int t = 0; t<10; t++)
    puzzle2DSeq(Nfil, Ncol, mZ, hX, hY);
  t2=GetTime();
  TiempoSEQ = (t2 - t1) / 10.0;
 
  // Obtener Memoria en el device
  cudaMalloc((float**)&dZ, numBytes); 
  cudaMalloc((float**)&dX, numBytes); 
  cudaMalloc((float**)&dY, numBytes); 
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(dX, hX, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dY, hY, numBytes, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  //
  // KERNEL POR FILAS 
  //

  nThreads = THREADS * THREADS;
  // nBlocks = Nfil/nThreads;  // Solo funciona bien si Nfil multiplo de nThreads
  nBlocks = (Nfil + nThreads - 1)/nThreads;  // Funciona bien en cualquier caso

  dim3 dimGridF(1, nBlocks, 1);
  dim3 dimBlockF(1, nThreads, 1);

  printf("\n");
  printf("\n");
  printf("\n");
  printf("INICIO PROGRAMA\n");
  printf("\n");
  printf("Kernel por Filas\n");
  printf("Dimension problema: %d filas x %d columnas\n", Nfil, Ncol);
  printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockF.x, dimBlockF.y, dimBlockF.z, dimBlockF.x * dimBlockF.y * dimBlockF.z);
  printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridF.x, dimGridF.y, dimGridF.z, dimGridF.x * dimGridF.y * dimGridF.z);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel por filas 
  puzzle2DPARfil<<<dimGridF, dimBlockF>>>(Nfil, Ncol, dZ, dX, dY);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en vZr para poder comprobar el resultado
  cudaMemcpy(hZ, dZ, numBytes, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Comprobar resultado
  if (test == 'Y') { if (Test2D(Nfil, Ncol, hZ, mZ)) printf ("FILAS - TEST PASS\n"); else printf ("FILAS - TEST FAIL\n"); }

  //
  // KERNEL POR COLUMNAS 
  //

  nThreads = THREADS * THREADS;
  // nBlocks = Ncol/nThreads;  // Solo funciona bien si Ncol multiplo de nThreads
  nBlocks = (Ncol + nThreads - 1)/nThreads;  // Funciona bien en cualquier caso

  dim3 dimGridC(nBlocks, 1, 1);
  dim3 dimBlockC(nThreads, 1, 1);

  printf("\n");
  printf("Kernel por Columnas\n");
  printf("Dimension problema: %d filas x %d columnas\n", Nfil, Ncol);
  printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockC.x, dimBlockC.y, dimBlockC.z, dimBlockC.x * dimBlockC.y * dimBlockC.z);
  printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridC.x, dimGridC.y, dimGridC.z, dimGridC.x * dimGridC.y * dimGridC.z);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  // Ejecutar el kernel por columnas 
  puzzle2DPARcol<<<dimGridC, dimBlockC>>>(Nfil, Ncol, dZ, dX, dY);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E4, 0);
  cudaEventSynchronize(E4);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en vZc para poder comprobar el resultado
  cudaMemcpy(hZ, dZ, numBytes, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Comprobar resultado
  if (test == 'Y') { if (Test2D(Nfil, Ncol, hZ, mZ)) printf ("COLUMNAS - TEST PASS\n"); else printf ("COLUMNAS - TEST FAIL\n"); }

  //
  // KERNEL ELEMENTO a ELEMENTO
  //

  nThreads = THREADS;
  int nBlocksFil = (Nfil+nThreads-1)/nThreads;
  int nBlocksCol = (Ncol+nThreads-1)/nThreads;

  //dim3 dimGridE(Ncol/nThreads, Nfil/nThreads, 1);
  dim3 dimGridE(nBlocksCol, nBlocksFil, 1);
  dim3 dimBlockE(nThreads, nThreads, 1);

  printf("\n");
  printf("Kernel Elemento a Elemento\n");
  printf("Dimension problema: %d filas x %d columnas\n", Nfil, Ncol);
  printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockE.x, dimBlockE.y, dimBlockE.z, dimBlockE.x * dimBlockE.y * dimBlockE.z);
  printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridE.x, dimGridE.y, dimGridE.z, dimGridE.x * dimGridE.y * dimGridE.z);

  cudaEventRecord(E5, 0);
  cudaEventSynchronize(E5);

  // Ejecutar el kernel elemento a elemento
  puzzle2DPAR1x1<<<dimGridE, dimBlockE>>>(Nfil, Ncol, dZ, dX, dY);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E6, 0);
  cudaEventSynchronize(E6);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en vZe para poder comprobar el resultado
  cudaMemcpy(hZ, dZ, numBytes, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Comprobar resultado
  if (test == 'Y') { if (Test2D(Nfil, Ncol, hZ, mZ)) printf ("ELEMENTOS - TEST PASS\n"); else printf ("ELEMENTOS - TEST FAIL\n"); }


  // Liberar Memoria del device 
  cudaFree(dX); cudaFree(dY); cudaFree(dZ);

  cudaDeviceSynchronize();


  cudaEventElapsedTime(&TiempoKfil, E1, E2);
  cudaEventElapsedTime(&TiempoKcol, E3, E4);
  cudaEventElapsedTime(&TiempoKele, E5, E6);
 
  cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaEventDestroy(E4); cudaEventDestroy(E5); cudaEventDestroy(E6);

  printf("\n");
  printf("Resumen Rendimiento\n");
  printf("Tiempo Paralelo Kernel filas: %4.6f ms (%4.2f GFLOPS)\n", TiempoKfil, ((float) 5*Nfil*Ncol) / (1000000.0 * TiempoKfil));
  printf("Tiempo Paralelo Kernel columnas: %4.6f ms (%4.2f GFLOPS)\n", TiempoKcol, ((float) 5*Nfil*Ncol) / (1000000.0 * TiempoKcol));
  printf("Tiempo Paralelo Kernel elemento a elemento: %4.6f ms (%4.2f GFLOPS)\n", TiempoKele, ((float) 5*Nfil*Ncol) / (1000000.0 * TiempoKele));
  printf("Tiempo Secuencial: %4.6f milseg (%4.2f GFLOPS)\n", TiempoSEQ, ((float) 5*Nfil*Ncol) / (1000000.0 * TiempoSEQ));


  if (PINNED) { cudaFreeHost(hZ); cudaFreeHost(mZ); cudaFreeHost(hX); cudaFreeHost(hY); }
  else { free(hZ); free(mZ); free(hX); free(hY); }

}


void InitM(int Nfil, int Ncol, float *M) {
   int i;
   for (i=0; i<Nfil*Ncol; i++) 
     M[i] = rand();
   
}
int error(float a, float b) {

  if (abs (a - b) / a > 0.000001) return 1;
  else  return 0;

}

int Test2D(int Nfil, int Ncol, float *vZ, float *hZ) {
   int i, j, ind;
   ind = 0;
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


