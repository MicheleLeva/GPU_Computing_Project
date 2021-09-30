#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../utils/common.h"

#define THREADS 32
#define BLOCKS 32
#define T 64

//TODO fare questa parametrica finchè il parallelismo è insufficiente 
//Capire cosa il paper intenda
//Merge all the subsequences produced in step 1 until the parallelism is insufficient.
__global__ void bitonic_warp_merge(int * keyin, int * output, int offset){
  
  int j = 0;
  int stage = 0;
  int k_0 = 0;
  int u = 0, index1 = 0, p = 0;
  float dim = 0;
  
  __shared__ int buffer[T];

  unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int subseq = blockIdx.x; //in quale warp siamo
  unsigned int start = offset * subseq; //primo elemento della sottosequenza (A e B) da riordinare

  int outIndex = start + threadIdx.x;
  int iA = start, iB = start + (offset / 2);
  int fA = start + (offset / 2), fB = start + offset;
  int tA = iA + threadIdx.x, tB = iB + threadIdx.x;
  bool compare;

  /*
  if (threadIdx.x == 0){
      printf("block %d - thread %d: subseq = %d, offset = %d\n", blockIdx.x, threadIdx.x, subseq, offset);
      printf("block %d - thread %d: iA = %d, fA = %d, iB = %d, fB = %d \n", blockIdx.x, threadIdx.x, iA, iB, fA, fB);
  }*/
    

  //printf("thread %d: tA = %d, tB = %d \n", threadIdx.x, tA, tB);

  //prendo prima sequenza di A e la prima di B e le copio sul buffer
  buffer[T/2 - 1 - threadIdx.x] = keyin[tA];
  buffer[T/2 + threadIdx.x] = keyin[tB];
  tA += THREADS;
  tB += THREADS;
  
  //A[3] < B[3]
  compare = buffer[0] < buffer[T - 1]; //se true, al prossimo caricamento prendo i primi T/2 valori di A

  int loops = 1;
  while(true)  {

    /*
    if (threadIdx.x == 0){
      printf("loop = %d,\nblock %d, thread %d, START of while: tA = %d, tB = %d \n", loops, blockIdx.x, threadIdx.x, tA, tB);
    }*/
    

    stage = 0;
    //bitonic based merge sort
    for(j = T/2; j>0; j/=2){ 
      
      dim = j * 2;
      if (dim < 2) dim = 2;
      u = ceil((threadIdx.x+1) * 2/dim); //indice della sottosequenza su cui il thread deve lavorare

      //printf("thread %d : u = %d \n", threadIdx.x, u);

      index1 = (u - 1) * dim; //primo indice della sottosequenza

      p = threadIdx.x - (u - 1) * (dim / 2); // posizione del thread nella sottosequenza simmetrica
          
      k_0 = index1 + p;

      /*
      if (threadIdx.x == 0){
        printf("block %d, thread %d : stage = %d, offset = %d \n", blockIdx.x, threadIdx.x, stage, j);
      }
      printf("block %d, thread %d : k_0 = %d \n", blockIdx.x, threadIdx.x, k_0);
      */
           
      //k0 ? position of preceding element in the thread's first pair to form ascending order
      if(buffer[k_0] > buffer[k_0 + j]){
          int tmp = buffer[k_0];
          buffer[k_0] = buffer[k_0 + j];
          buffer[k_0 + j] = tmp;
      }

      stage++;
    }
    
    //carico i primi T/2 elementi di buffer sull'output
    output[outIndex] = buffer[threadIdx.x];
    outIndex += THREADS;

    //se A e B finiscono elementi prima dell'algoritmo, prosegui solo con la sottosequenza rimanente
    if (tA > fA - 1 && tB < fB - 1)
      compare = false;
    if (tA < fA - 1 && tB > fB - 1)
      compare = true;
    if (tA > fA - 1 && tB > fB - 1){
        
      //carico gli ultimi T/2 elementi del buffer sull'output
      output[outIndex] = buffer[T/2 + threadIdx.x];
      
      break;
    }
      
    
    //usa il compare per caricare la prossima sottosequenza da A o B   
    if (compare){
      //carico T/2 elementi da A al buffer
      buffer[T/2 - 1 - threadIdx.x] = keyin[tA];
      tA += THREADS;
    } else {
      //carico T/2 elementi da B al buffer
      buffer[T/2 - 1 - threadIdx.x] = keyin[tB];
      tB += THREADS;
    }

    if (compare){ //se avevo caricato dalla sequenza A, allora Amax è il primo elemento del buffer e Bmax è l'ultimo
        compare = buffer[0] < buffer[T - 1];
    } else { //altrimenti ho caricato B sul buffer, e Amax è l'ultimo elemento, mentre Bmax è il primo
        compare = buffer[0] > buffer[T - 1];
    }

    loops++;

    /*
    if (threadIdx.x == 0){
      printf("thread %d, END of while: tA = %d, tB = %d \n",threadIdx.x, tA, tB);
    }*/
  }

}

/*Divide the input sequence into equal-sized subsequences. 
  Each subsequence will be sorted by an independent warp using the bitonic network.*/
__global__ void bitonic_sort_warp(int *keyin){
  //prendere thread id giusto tenendo in considerazione k_0 e k_1
  //implementare gli swap fatti bene dentro la funzione
  unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int subseq = id / 32; //in quale sottosequenza dell'array siamo
  unsigned int start = 128 * subseq; //primo elemento della sottosequenza da riordinare

  int i = 0, j = 0;
  int phase = 0, stage = 0;
  int k_0 = 0, k_1 = 0;
  int u = 0, index1 = 0, index2 = 0, p = 0, q = 0, m = 0, o = 0, um = 0, pm = 0;
  float dim = 0;

  //phase 0 to log(128)-1 
  for(i=2; i<128 ;i*=2){ 
    stage = 0;


    dim = i*2;
    u = ceil( (threadIdx.x+1) * (4/dim) ); //indice della sottosequenza simmetrica a cui il thread appartiene
    //printf("thread %d : u = %d \n", threadIdx.x, u);

    index1 = (u - 1) * dim;
    index2 = index1 + dim - 1;

    for(j = i/2; j > 0; j /= 2){ 
      /*
      if (threadIdx.x == 0)
        printf("thread %d : phase = %d, stage = %d \n", threadIdx.x, phase, stage);
      */
      p = threadIdx.x - (u - 1) * (dim / 4); // posizione del thread nella sottosequenza simmetrica
      //printf("thread %d : p = %d \n", threadIdx.x, p);

      //q è l'offset usato poi per k_0 e k_1
      if (stage == 0) { // primo stage della fase
          q = p;
      }
      if (stage != 0 && stage != phase){ //né primo né ultimo stage della fase
          
          //int n = 2 ^ stage; // numero di minisequenze
          m = j; // numero di freccie rosse per minisequenza
          o = j * 2; //offset speciale tra minisequenza e l'altra

          um = (int)(p / m); //indice della minisequenza a cui il thread appartiene

          pm = p - um * m; //posizione del thread nella minisequenza
          q = pm + o * um;
      }
      if (stage == phase){ //ultimo stage della fase
          q = p * 2;
      }
      k_0 = index1 + q;
      k_1 = index2 - q; 

      k_0 = start + k_0;
      k_1 = start + k_1; 
      
      //printf("thread %d : k_0 = %d, k_1 = %d \n", threadIdx.x, k_0, k_1);

      //k_0 ? position of preceding element in each pair to form ascending order
      if(keyin[k_0] > keyin[k_0+j]) {
        int tmp = keyin[k_0];
        keyin[k_0] = keyin[k_0+j];
        keyin[k_0+j] = tmp;
      }
      //k1 ? position of preceding element in each pair to form descending order
      if(keyin[k_1] > keyin[k_1-j]){
        int tmp = keyin[k_1];
        keyin[k_1] = keyin[k_1-j];
        keyin[k_1-j] = tmp;
      }

      stage++;
    }
    phase++;
  }

  stage = 0;
  //special case for the last phase 
  for(j=128/2; j>0; j/=2){
    
    dim = j * 2;
    if (dim < 4) dim = 4;
    u = ceil( (threadIdx.x+1) * (4/dim) ); //indice della sottosequenza simmetrica a cui il thread appartiene

    //printf("thread %d : u = %d \n", threadIdx.x, u);

    index1 = (u - 1) * dim;
    index2 = index1 + dim - 1;

    p = threadIdx.x - (u - 1) * (dim / 4); // posizione del thread nella sottosequenza simmetrica

    //q è l'offset usato poi per k_0 e k_1
    
    q = p;
        
    k_0 = index1 + q;
    k_1 = index2 - q; 

    k_0 = start + k_0;
    k_1 = start + k_1;

    /*
    if (threadIdx.x == 0)
        printf("thread %d : stage = %d, offset = %d \n", threadIdx.x, stage, j);
    printf("thread %d : k_0 = %d, k_1 = %d \n", threadIdx.x, k_0, k_1);
    */
      
    //k0 ? position of preceding element in the thread's first pair to form ascending order
    if(keyin[k_0] > keyin[k_0+j]){
        int tmp = keyin[k_0];
        keyin[k_0] = keyin[k_0 + j];
        keyin[k_0 + j] = tmp;
    }

    //k1 ? position of preceding element in the thread's second pair to form ascending order
    if(keyin[k_1] < keyin[k_1 - j]){
        int tmp = keyin[k_1];
        keyin[k_1] = keyin[k_1 - j];
        keyin[k_1 - j] = tmp;
    }

    stage++;
  }
}



/*The parameter dir indicates the sorting direction, ASCENDING
 or DESCENDING; if (a[i] > a[j]) agrees with the direction,
 then a[i] and a[j] are interchanged.*/
void compAndSwap(int a[], int i, int j, int dir) {
	if (dir == (a[i] > a[j])) {
		int tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
	}
}

/*It recursively sorts a bitonic sequence in ascending order,
 if dir = 1, and in descending order otherwise (means dir=0).
 The sequence to be sorted starts at index position low,
 the parameter cnt is the number of elements to be sorted.*/
void bitonicMerge(int a[], int low, int cnt, int dir) {
	if (cnt > 1) {
		int k = cnt / 2;
		for (int i = low; i < low + k; i++)
			compAndSwap(a, i, i + k, dir);
		bitonicMerge(a, low, k, dir);
		bitonicMerge(a, low + k, k, dir);
	}
}

/* This function first produces a bitonic sequence by recursively
 sorting its two halves in opposite sorting orders, and then
 calls bitonicMerge to make them in the same order */
void bitonicSort(int a[], int low, int cnt, int dir) {
	if (cnt > 1) {
		int k = cnt / 2;

		// sort in ascending order since dir here is 1
		bitonicSort(a, low, k, 1);

		// sort in descending order since dir here is 0
		bitonicSort(a, low + k, k, 0);

		// Will merge wole sequence in ascending order
		// since dir=1.
		bitonicMerge(a, low, cnt, dir);
	}
}

/*
 * test bitonic sort on CPU and GPU
 */
int main(void) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int N = THREADS*4*BLOCKS;
	// check
	if (!(N && !(N & (N - 1)))) {
		printf("ERROR: N must be power of 2 (N = %d)\n", N);
		exit(1);
	}
	size_t nBytes = N * sizeof(int);
	int *a = (int*) malloc(nBytes);
	int *b = (int*) malloc(nBytes);

	// fill data
	for (int i = 0; i < N; ++i) {
		//a[i] =  i%5; //rand() % 100; // / (float) RAND_MAX;
    a[i] = rand() % 100;
		b[i] = a[i];
	}

	// bitonic CPU
	double cpu_time = seconds();

  bitonicSort(b, 0, N, 1);   // 1 means sort in ascending order

	printf("CPU elapsed time: %.5f (sec)\n", seconds()-cpu_time);

	// device mem copy
	int *d_a, * d_b;
	CHECK(cudaMalloc((void**) &d_a, nBytes));
  CHECK(cudaMalloc((void**) &d_b, nBytes));
	CHECK(cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice));

	// num of threads
	dim3 blocks(BLOCKS, 1);   // Number of blocks
  dim3 threads(THREADS, 1); // Number of threads
	
  /*
	int j, k;
  // external loop on comparators of size k
  for (k = 2; k <= N; k <<= 1) {
    // internal loop for comparator internal stages
    for (j = k >> 1; j > 0; j = j >> 1)
      bitonic_sort_step<<<blocks, threads * 4>>>(d_a, j, k);
  }
  */
	
  // start computation
	cudaEventRecord(start);
  /*Divide the input sequence into equal-sized subsequences. 
  Each subsequence will be sorted by an independent warp using the bitonic network.*/
  bitonic_sort_warp<<<blocks, threads>>>(d_a);

  cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
  int l = deviceProp.multiProcessorCount; //numero di streaming multiprocessor della GPU

  printf ("Streaming multiprocessors = %d\n", l);

 
  /*Merge all the subsequences produced in step 1 until the parallelism is insufficient.*/
  //finchè il parallelismo è insufficiente
  //ovvero finchè N / offset >= l

  //bitonic_warp_merge<<<blocks, threads>>>(d_a, d_b, 256);
  //blocks.x = blocks.x / 2;   // Number of blocks
  //bitonic_warp_merge<<<blocks, threads>>>(d_a, d_a, 512);

  //ad ogni warp merge si inverte input ed output
  bool isAfirst = true;
  blocks.x = BLOCKS / 2;   // Number of blocks
  for(int offset = THREADS * 8; N / offset >= 1; offset *= 2){
    if(isAfirst)
      bitonic_warp_merge<<<blocks, threads>>>(d_a, d_b, offset);
    else
      bitonic_warp_merge<<<blocks, threads>>>(d_b, d_a, offset);
    blocks.x = blocks.x / 2;
    
    isAfirst = !isAfirst;
  }
  //TODO ricordarsi che Afirst è poi true se l'output finale è in A, false se è in B
  //TODO ricorarsi di resettare il numero di blocchi (warp)
 

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

	// recover data
  if (isAfirst)
	  cudaMemcpy(a, d_a, nBytes, cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(a, d_b, nBytes, cudaMemcpyDeviceToHost);

	// print & check
	if (N < 1000) {
		printf("GPU:\n");
		for (int i = 0; i < N; ++i){
      if(i % 128 == 0)
        printf("sottosequenza, indice = %d\n", i);
      printf("%d : %d\n", i, a[i]);
    }
			
      /*
		printf("CPU:\n");
		for (int i = 0; i < N; ++i)
			printf("%d\n", b[i]);
      */
	}
	else {
    
		for (int i = 0; i < N; ++i) {
			if (a[i] != b[i]) {
				printf("ERROR a[%d] != b[%d]  (a[i] = %d  -  b[i] = %d\n", i,i, a[i],b[i]);
				break;
			}
		}
	}

	cudaFree(d_a);
	exit(0);
}