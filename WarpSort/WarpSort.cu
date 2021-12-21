#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../utils/common.h"

#define THREADS 32
#define BLOCKS 64
#define T 64
#define K 8

int * get_splitters (int * input, int N, int s); //preliminary

__global__ void bitonic_sort_warp(int *keyin); //step1

__global__ void bitonic_warp_merge(int * keyin, int * output, int offset); //step2

__global__ void print_array_kernel(int * input, int length);

__global__ void printMatrix(int * matrix, int * a, int rows, int columns);

__global__ void load_placeholders(int * s_lengths, int * d_buffer, int * a, int * s_indexesByColumn);

__global__ void loadOutput(int * d_buffer, int * d_b, int l, int columnLength, int global_index);

__global__ void loadSplitterMatrix(int * gpu_indexMatrix, int * splitters, int * d_a, int N, int l, int s);

int get_length (int * array);

void shuffle(int *array, size_t n) {    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int usec = tv.tv_usec;
    srand48(usec);


    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
            size_t j = (unsigned int) (drand48()*(i+1));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

/******FUNCTIONS*****/



//kernel che stampa i contenuti dell'array in input
__global__ void print_array_kernel(int * input, int length){
    for(int i = 0; i < length; i++){
        printf("Array[%d] = %d \n", i, input[i]);
    }
}

//kernel che stampa una matrice
__global__ void printMatrix(int * matrix, int * a, int rows, int columns){
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            printf("matrice[l = %d, s = %d] = %d -> valore = %d\n", i, j, matrix[columns * i + j], a[matrix[columns * i + j]]);
        }
    }
}

//Preliminary splitter preparation function
int * get_splitters (int * input, int N, int s){
  int numElements = s * K;

  //printf("numElements = %d\n", numElements);

  size_t nBytes = numElements * sizeof(int);
	int *arrayA = (int*) malloc(nBytes);

  //seleziona a random numElements elementi da input
  for (int i = 0; i < numElements; i++){
      arrayA[i] = input[rand() % numElements];
      //printf("arrayA[%d] = %d\n", i, arrayA[i]);
  }

  // num of threads
	dim3 blocks(numElements / 128, 1);   // Number of blocks
  dim3 threads(THREADS, 1); // Number of threads

  //device memcopy
  int *bufferA, *bufferB;
  CHECK(cudaMalloc((void**) &bufferA, nBytes));
  CHECK(cudaMalloc((void**) &bufferB, nBytes));
	CHECK(cudaMemcpy(bufferA, arrayA, nBytes, cudaMemcpyHostToDevice));
  free(arrayA);

  //printf("\nbufferA prima step1 SORT\n\n");
  //print_array_kernel<<<1, 1>>>(bufferA, numElements);

  //sorting degli elementi

  bitonic_sort_warp<<<blocks, threads>>>(bufferA);

  //printf("\nbufferA dopo step1 SORT\n\n");
  //print_array_kernel<<<1, 1>>>(bufferA, numElements);

  bool isAfirst = true;
  if(numElements > 128){
    //ad ogni warp merge si inverte input ed output
    blocks.x = (numElements / 128) / 2;   // Number of blocks
    for(int offset = THREADS * 8; numElements / offset >= 1; offset *= 2){
      //printf("N = %d, offset = %d, blocks.x = %d, threads.x = %d\n", N, offset, blocks, threads);
      if(isAfirst)
        bitonic_warp_merge<<<blocks, threads>>>(bufferA, bufferB, offset);
      else
        bitonic_warp_merge<<<blocks, threads>>>(bufferB, bufferA, offset);
      blocks.x = blocks.x / 2;
      
      isAfirst = !isAfirst;
    }
  }
  
  //printf("\nbufferA dopo SORT\n\n");
  //print_array_kernel<<<1, 1>>>(bufferA, numElements);

  //printf("\nbufferB dopo SORT\n\n");
  //print_array_kernel<<<1, 1>>>(bufferB, numElements);

  int * orderedSequence = (int*) malloc(numElements * sizeof(int));
  if(isAfirst){
    cudaMemcpy(orderedSequence, bufferA, nBytes, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(orderedSequence, bufferB, nBytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(bufferA);
  cudaFree(bufferB);

  /*
  for (int i = 0; i < numElements; i++){
      printf("orderedSequence[%d] = %d\n", i, orderedSequence[i]);
  }*/

  //seleziona k elementi dal buffer ordinato e restituisci
  int *output = (int*) malloc(s*sizeof(int));
  int last_split = -1;
  for(int i = 0; i < s; i ++){
      output[i] = orderedSequence[i*K];
      if (output[i] == last_split){
          output[i]++;
      }
      last_split = output[i];
      //printf("output[%d] = %d\n", i, output[i]);
  }

  free(orderedSequence);
   
  return output;

}

/*STEP 1:  Divide the input sequence into equal-sized subsequences. 
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

  //if (threadIdx.x == 0) printf("bitonic_sort_warp\n");

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

//STEP 2: Merge all the subsequences produced in step 1 until the parallelism is insufficient.
__global__ void bitonic_warp_merge(int * keyin, int * output, int offset){
  
  int j = 0;
  int stage = 0;
  int k_0 = 0;
  int u = 0, index1 = 0, p = 0;
  float dim = 0;
  
  __shared__ int buffer[T];

  //unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int subseq = blockIdx.x; //in quale warp siamo
  unsigned int start = offset * subseq; //primo elemento della sottosequenza (A e B) da riordinare

  //if (threadIdx.x == 0) printf("bitonic_warp_merge, offset = %d\n", offset);

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

      
      if (threadIdx.x == 0){
        //printf("block %d, thread %d : stage = %d, offset = %d \n", blockIdx.x, threadIdx.x, stage, j);
      }
      //printf("block %d, thread %d : k_0 = %d \n", blockIdx.x, threadIdx.x, k_0);
      
           
      //k0 ? position of preceding element in the thread's first pair to form ascending order
      if(buffer[k_0] > buffer[k_0 + j]){
          int tmp = buffer[k_0];
          buffer[k_0] = buffer[k_0 + j];
          buffer[k_0 + j] = tmp;
      }

      stage++;
    }
    
    //carico i primi T/2 elementi di buffer sull'output
    if (outIndex > fB)
      printf("ERRORE! outIndex è troppo grande -> outIndex = %d > tA = %d, fA = %d, tb = %d, fB = %dt\n", outIndex,
             tA, fA, tB, fB);
    else
      output[outIndex] = buffer[threadIdx.x];
    outIndex += THREADS;

    //se A e B finiscono elementi prima dell'algoritmo, prosegui solo con la sottosequenza rimanente
    if (tA >= fA  && tB < fB )
      compare = false;
    if (tA < fA && tB >= fB)
      compare = true;
    if (tA >= fA && tB >= fB){
        
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

//STEP 3:
__global__ void loadSplitterMatrix(int * gpu_indexMatrix, int * splitters, int * d_a, int N, int l, int s){

  int rowLength = N / l;
  int row = blockIdx.x;
  int splitterIndex = threadIdx.x;

  extern __shared__ int sharedRow[];

  int len = rowLength / s;

  if (blockIdx.x == 0 && threadIdx.x == 0)
    printf("len = rowLength / s = %d, rowlength = %d\n", len, rowLength);

  if (blockIdx.x == 1 && threadIdx.x == 0)
    printf("len = rowLength / s = %d, rowlength = %d, splitterIndex = %d, row = %d, s = %d\n", len, rowLength, splitterIndex, row, s);

  /*printf("sono il thread %d, %d e copio da d_a[%d * %d + %d * %d] a sharedRow[%d * %d]\n", row, splitterIndex, 
         rowLength, row, splitterIndex, len, splitterIndex, len);*/
  //caricamento della row nella memoria shared
  for (int i = 0; i < len; i++){
      sharedRow[splitterIndex * len + i] = d_a[rowLength * row + splitterIndex * len + i];
  }

  /*
  if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i = 0; i < rowLength; i++){
          printf("sharedRow[%d] = %d\n", i, sharedRow[i]);
      }
  }*/

  
  if(splitterIndex == 0){
      
    //caricamento dell'indice corrispondente allo splitter sulla matrice di output (caso primo indice della riga)
    gpu_indexMatrix[s * row] = rowLength * row;

  } else if (splitterIndex < s && splitterIndex > 0) {
      
    //ricerca dell'indice corrispondente allo splitter
    int i;
    for (i = 0; (splitters[splitterIndex] > sharedRow[i]) && i < len ; i++) {;}
    
    //caricamento dell'indice corrispondente allo splitter sulla matrice di output
    gpu_indexMatrix[s * row + splitterIndex] = rowLength * row + i;
  }
  
}


//Used in step 4 to load -1
__global__ void load_placeholders(int * s_lengths, int * d_buffer, int * a, int * s_indexesByColumn){
    
  unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int subseq = blockIdx.x; //which segment the kernel is working on
  int index = s_indexesByColumn[subseq] + threadIdx.x - (128 - s_lengths[subseq]);

  /*
  if (threadIdx.x == 0)
    printf("Load_placeholders subseq: %d , threadIdx.x = %d, s_indexesByColumn = %d, s_lengths[subseq] = %d\n",
           subseq, threadIdx.x, index, s_lengths[subseq]);
  //printf("Load_placeholders subseq: %d , id = %d, index = %d\n", subseq, id, index);
  */

  if (threadIdx.x < 128 - s_lengths[subseq])
    d_buffer[id] = -1;
  else{
    d_buffer[id] = a[index];
  }
    
}

__global__ void loadOutput(int * d_buffer, int * d_b, int l, int columnLength, int global_index){

  unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
  int local_index = id - ( l * 128 - columnLength);
  int index = global_index + local_index;
  /*
  if (local_index == -1)
        printf("Load_output local index = -1, id = %d, l = %d, columnLength = %d, global_index = %d, index = %d, d_buffer[%d] = %d\n", 
              id, l, columnLength, global_index, index, id, d_buffer[id]);*/

  if (id >= (l * 128 - columnLength)){
    /*
    if (local_index == 0)
      printf("Load_output local index = 0, id = %d, l = %d, columnLength = %d, global_index = %d, index = %d, d_buffer[%d] = %d\n", 
             id, l, columnLength, global_index, index, id, d_buffer[id]);
    if (local_index == columnLength - 1)
      printf("Load_output local index = columnLength - 1, id = %d, l = %d, columnLength = %d, global_index = %d, index = %d, d_buffer[%d] = %d\n", 
             id, l, columnLength, global_index, index, id, d_buffer[id]);
    */
    //printf("dbuffer[%d] = %d\n", id, d_buffer[id]);
    d_b[index] = d_buffer[id];
  }
    
  
}


/*******FUNZIONI DEL PROFESSORE*********/

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
 ******************* MAIN *****************************************************************
 */
int main(void) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int N = THREADS*4*BLOCKS;
  printf ("N = %d\n",N);
	// check
	if (!(N && !(N & (N - 1)))) {
		printf("ERROR: N must be power of 2 (N = %d)\n", N);
		exit(1);
	}
	size_t nBytes = N * sizeof(int);
	int *a = (int*) malloc(nBytes);
	int *b = (int*) malloc(nBytes);

  srand ( time(NULL) );
	// fill data
	for (int i = 0; i < N; ++i) {
		//a[i] =  i%5; //rand() % 100; // / (float) RAND_MAX;
    //a[i] = rand() % 1000;
    a[i] = i;
	}
  
  shuffle(a, N);

  for (int i = 0; i < N; i++){
      b[i] = a [i];
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
	
  cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
  int l = deviceProp.multiProcessorCount; //numero di streaming multiprocessor della GPU
  for (int k = 2; k < 1000; k *= 2){ 
      if (k > l){ //TODO controllare se moltiplicando per 4 come suggerisce il paper migliora le prestazioni
          l = k;
          break;
      }
  }
  int s = BLOCKS / 2;

  printf ("\nStreaming multiprocessors = %d\n", l);
	
  cudaEvent_t start_step, stop_step;
	cudaEventCreate(&start_step);
  cudaEventCreate(&stop_step);
  float milliseconds_step = 0;
  

  // start computation
	cudaEventRecord(start);

  /*PRELIMINARY SPLITTER STEP3*********************************************************************/
  //si trovano gli splitter che poi verranno usati nello step 3
  printf("\n*****PRELIMINARY STEP*****\n");
  cudaEventRecord(start_step);

  int *output = get_splitters (a, N, s);

  cudaEventRecord(stop_step);
	cudaEventSynchronize(stop_step);
  cudaEventElapsedTime(&milliseconds_step, start_step, stop_step);
  printf("Preliminary step time: %.5f (sec)\n", milliseconds_step / 1000);

  /*
  //Check se il sort del preliminary step è avvenuto correttamente
  for (int i = 1; i < s; i++){
      if (output[i] < output[i-1])
        printf("Preliminary step: errore! -> output[%d] = %d < output[%d] = %d\n", i, i-1, output[i], output[i-1]);
  }*/
  
  /*STEP 1: Divide the input sequence into equal-sized subsequences. *******************************************
  Each subsequence will be sorted by an independent warp using the bitonic network.*/
  printf("\n*****STEP1*****\n");

  cudaEventRecord(start_step);

  bitonic_sort_warp<<<blocks, threads>>>(d_a);

  cudaEventRecord(stop_step);
	cudaEventSynchronize(stop_step);
  cudaEventElapsedTime(&milliseconds_step, start_step, stop_step);
  printf("Step 1 time: %.5f (sec)\n", milliseconds_step / 1000);

  /*//Check se il sort dello step 1 è avvenuto correttamente
  int* temp = (int*) malloc(nBytes);
  CHECK(cudaMemcpy(temp, d_a, nBytes, cudaMemcpyDeviceToHost));
  for (int i = 1; i < N; i++){
      if ((temp[i] < temp[i-1]) && (i % 128 != 0))
        printf("Step 1: errore! -> temp[%d] = %d < temp[%d] = %d\n", i, i-1, temp[i], temp[i-1]);
  }*/

  
  /*STEP 2: Merge all the subsequences produced in step 1 until the parallelism is insufficient.*******************/
  bool isAfirst = true;
  blocks.x = BLOCKS / 2;   // Number of blocks

  printf("\n*****STEP2*****\n");

  cudaEventRecord(start_step);

  int maxOrderedSegmentSize;
  //finchè il parallelismo è insufficiente, ovvero finchè N / offset >= l
  //ad ogni warp merge si inverte input ed output raddoppiando la lunghezza del segmento che il warp deve ordinare
  for(int offset = THREADS * 8; N / offset >= l; offset *= 2){
    //printf("Step 2 - Offset = %d\n", offset); 
    if(isAfirst)
      bitonic_warp_merge<<<blocks, threads>>>(d_a, d_b, offset);
    else
      bitonic_warp_merge<<<blocks, threads>>>(d_b, d_a, offset);
    blocks.x = blocks.x / 2;
    
    isAfirst = !isAfirst;
    maxOrderedSegmentSize = offset;
  }
  

  //recover data
  if(!isAfirst){
      CHECK(cudaMemcpy(a, d_b, nBytes, cudaMemcpyDeviceToHost));
  } else {
      CHECK(cudaMemcpy(a, d_a, nBytes, cudaMemcpyDeviceToHost));
  }

  cudaEventRecord(stop_step);
	cudaEventSynchronize(stop_step);
  cudaEventElapsedTime(&milliseconds_step, start_step, stop_step);
  printf("Step 2 time: %.5f (sec)\n", milliseconds_step / 1000);

  /*
  //Check se il sort dello step 2 è avvenuto correttamente
  for (int i = 1; i < N; i++){
      if ((a[i] < a[i-1]) && (i % maxOrderedSegmentSize != 0))
        printf("Step 2: errore! -> a[%d] = %d < a[%d] = %d\n", i, i-1, a[i], a[i-1]);
  }*/

  /*
  for (int i = 0; i < 2048; i++){
    printf("a[%d] = %d\n", i, a[i]);
  }*/

  
  /*STEP 3: Split the large subsequences produced in step 2 into small ones that can be merged independently.*******************/

  printf("\n*****STEP3*****\n");

  cudaEventRecord(start_step);

  int s_indexes[l][s];

  /*
  int index;
  for (int i = 0; i < l; i++){
    for (int j = 0; j < s; j++){
      if (j == 0){
          s_indexes[i][j] = N / l * i;
      }
      else if (j < s){
        index = N / l * i + 1;
        while (a[index] < output[j]){
            index++;
        }
        s_indexes[i][j] = index;
      }
    }
  }*/
  
  //versione GPU 
  //TODO SISTEMARE, usare stream forse???
  //allocazione della matrice degli indici
  int* gpu_indexMatrix;
  CHECK(cudaMalloc((void**) &gpu_indexMatrix, s * l * sizeof(int)));

  int* splitters;
  CHECK(cudaMalloc((void**) &splitters, s * sizeof(int)));
  CHECK(cudaMemcpy(splitters, output, sizeof(int) * s, cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice));

  printf("prima di loadSplitterMatrix\n");
  int rowLength = N / l; 
  loadSplitterMatrix<<<l, s, rowLength * sizeof(int)>>>(gpu_indexMatrix, splitters, d_a, N, l, s);
  printf("dopo loadSplitterMatrix\n");

  printMatrix<<<1, 1>>>(gpu_indexMatrix, d_a, l, s);

  //registrazione del tempo step 3
  cudaEventRecord(stop_step);
	cudaEventSynchronize(stop_step);
  cudaEventElapsedTime(&milliseconds_step, start_step, stop_step);
  printf("Step 3 time: %.5f (sec)\n", milliseconds_step / 1000);

  /*
  for (int i = 0; i < l; i++){
      for (int j = 0; j < s; j++){
          //printf("indice %d, %d di l, s = %d -> valore = %d\n", i, j, s_indexes[i][j], a[s_indexes[i][j]]);
          if (j - 1 >= 0) {
              if (a[s_indexes[i][j]] < output[j] ) printf("Errore: primo valore del segmento %d, %d %d è minore dello splitter %d\n", i, j, 
                                                                         a[s_indexes[i][j]], output[j]);
          } else {
              if (a[N / l * i] < 0) printf("Errore: primo valore del segmento %d, %d %d è minore di zero!\n", i, j, 
                                                                         a[N / l * i]);
          }
          
          if (j + 1 < s){
            if (a[s_indexes[i][j + 1] - 1] >= output[j + 1]) printf("Errore: ultimo valore del segmento %d, %d %d è maggiore o uguale allo splitter %d\n", i, j, 
                                                                         a[s_indexes[i][j] - 1], output[j + 1]); 
          } else {
            if (a[N / l * i - 1] >= N) printf("Errore: ultimo valore del segmento %d, %d %d è maggiore o uguale allo splitter %d e ha sforato N!\n", i, j, 
                                                                         a[N / l * i - 1], N);
          }
      }
  }*/

  //TODO rimuovere
  return 0;

  /****STEP 4: *************************************************************************************************/
  
  cudaEventRecord(start_step);

  int global_index = 0;
  int s_lengths[l], s_indexesByColumn[l];
  int global_s_lengths = 0;

  int *a_output;
  a_output = (int*) malloc(nBytes);
  CHECK(cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, a, nBytes, cudaMemcpyHostToDevice));

  printf("\n*****STEP4*****\n");

  //TODO (forse) possibile parallelizzazione pure in questo caso? da lavoro su una colonna alla volta a tutta la matrice in una volta + rimozione padding?
  for (int i = 0; i < s; i++){ //per ogni colonna
    int *cpu_buffer; //buffer on cpu used to build the first s segment with -1 placeholders
    cpu_buffer = (int*) malloc(l * 128 * sizeof(int));
    int columnLength = 0;

    int *d_buffer, *d_buffer_temp;
    CHECK(cudaMalloc((void**) &d_buffer, l * 128 * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_buffer_temp, l * 128 * sizeof(int)));

    //TODO portare sulla GPU per sfruttare la matrice degl indici GPU
    //copia delle lunghezze dei segmenti s in un buffer s_lengths e controllo che non sforino i 128 elementi
    for (int j = 0; j < l; j++){
      if (i + 1 >= s){
        if (j + 1 >= l)
          s_lengths[j] = N - s_indexes[j][i]; //caso limite ultimo segmento
        else
          s_lengths[j] = s_indexes[j + 1][0] - s_indexes[j][i]; //ultimo segmento della riga
      } else{
        s_lengths[j] = s_indexes[j][i + 1] - s_indexes[j][i]; //calcoliamo la lunghezza del segmento s
      }

      s_indexesByColumn[j] = s_indexes[j][i];

      //printf("segmento %d, %d: s_length = %d\n", j, i, s_lengths[j]);
      if(s_lengths[j] > 128){
          printf("\n\n ERRORE SEGMENTO > 128\n\n");
          break;
      }

      columnLength += s_lengths[j];
      //global_s_lengths += s_lengths[j];
    }

    int * gpu_colIndexes, *gpu_s_lengths;
    CHECK(cudaMalloc((void**) &gpu_colIndexes, sizeof(int) * l));
    CHECK(cudaMalloc((void**) &gpu_s_lengths, sizeof(int) * l));
    CHECK(cudaMemcpy(gpu_colIndexes, s_indexesByColumn, sizeof(int) * l, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_s_lengths, s_lengths, sizeof(int) * l, cudaMemcpyHostToDevice));
    
    //caricamento dei segmenti sul buffer d_buffer assieme ai placeholder -1
    load_placeholders<<<l, 128>>>(gpu_s_lengths, d_buffer, d_a, gpu_colIndexes);
    
    //CHECK(cudaMemcpy(d_buffer, cpu_buffer, l * 128 * sizeof(int), cudaMemcpyHostToDevice));

    //riordino di d_buffer (colonna) attraverso bitonic_warp_merge come in step 2
    blocks.x = l / 2;   // Number of blocks (warps)
    isAfirst = true;
    for(int offset = THREADS * 8; l * 128 / offset >= 1 ; offset *= 2){ 
      //printf("Step 2 nello step4: offset = %d, blocks.x = %d\n", offset, blocks.x );
      if(isAfirst)
        bitonic_warp_merge<<<blocks, threads>>>(d_buffer, d_buffer_temp, offset);
      else
        bitonic_warp_merge<<<blocks, threads>>>(d_buffer_temp, d_buffer, offset);
      blocks.x = blocks.x / 2;
      
      isAfirst = !isAfirst;
    }
    
    //copia della colonna riordinata su output d_b
    if(!isAfirst){
      loadOutput<<<l, 128>>>(d_buffer_temp, d_b, l, columnLength, global_index);
    } else {
      loadOutput<<<l, 128>>>(d_buffer, d_b, l, columnLength, global_index);
    }
    global_index += columnLength;
   
    //printf("global_index nel for, colonna %d = %d\n", i, global_index);
    //printf("num_veri nel for, colonna %d = %d\n", i, num_veri);
    //printf("global_s_lengths nel for, colonna %d = %d\n", i, global_s_lengths);
    
    cudaFree(d_buffer); 
    cudaFree(d_buffer_temp); 
    free(cpu_buffer);
  }

  //printf("\n\nglobal_index = %d\n", global_index);
  //printf("global_s_lengths = %d\n", global_s_lengths);
  
	// recover data
  cudaMemcpy(a_output, d_b, nBytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop_step);
	cudaEventSynchronize(stop_step);
  cudaEventElapsedTime(&milliseconds_step, start_step, stop_step);
  printf("Step 4 time: %.5f (sec)\n", milliseconds_step / 1000);

  //total elapsed time print
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

	// print & check

  bool errors = false;

	if (N < 100) {
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
      //printf("a[%d] = %d, b[%d] = %d)\n", i, a_output[i], i, b[i]);
			if (a_output[i] != b[i]) {
				printf("ERROR a[%d] != b[%d]  (a[i] = %d  -  b[i] = %d)\n", i,i, a_output[i],b[i]);
        errors = true;
				break;
			}
		}
	}

  if (!errors)
    printf("SORTING AVVENUTO CON SUCCESSO!\n");

  free(a);
  free(a_output);
	cudaFree(d_a);
  cudaFree(d_b);
	exit(0);
}