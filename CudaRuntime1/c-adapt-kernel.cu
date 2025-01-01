#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <openssl/sha.h>
#include <string>
#include <vector>
#include "C-SHA-3/sha3.cuh"


//TODO convert sha3 etc to .cu 

__global__ void
bruteSearch(char** ptr_hash, char*** ptr_wordlist, size_t n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x; //blockDim.x threads per block
	if (index < n) {
		sha3_context c;
		//sha3_Init256(&c);
		//const void* hash = sha3_Finalize(&c);
		//printf("index %d", hash);
	}
}


cudaError_t loadParallelHashes(char* desiredHash, size_t unhashed_size, char** wordlist, size_t wordlistLength) //todo array decay?
{
#define THREADS_PER_BLOCK 512;
	char** dev_desiredHash = 0;
	char*** dev_wordlist = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//malloc and cpy for desired hash
	cudaStatus = cudaMalloc((void**)&dev_desiredHash, unhashed_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_desiredHash, &desiredHash, unhashed_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//malloc and cpy for wordlist
	cudaStatus = cudaMalloc((void****)&dev_wordlist, wordlistLength);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_wordlist, &wordlist, wordlistLength, cudaMemcpyHostToDevice);

	int numBlocks = wordlistLength / THREADS_PER_BLOCK;

	//bruteSearch<<<(wordlistLength/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dev_desiredHash, dev_wordlist);
	bruteSearch << <1, 512 >> > (dev_desiredHash, dev_wordlist, wordlistLength);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch. (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}


Error:
	return cudaStatus;
}

int main(int argc, char* argv[]) {
	int numThreads;
	if (argc == 2) {
		numThreads = atoi(argv[1]);
	}
	else {
		numThreads = 2;
	}



	std::string curLine;
	std::ifstream stream("file.txt");

	std::vector<std::string> v;

	int numLines = 0;
	while (std::getline(stream, curLine)) {
		numLines++;
		v.push_back(curLine);
	}

	char** charArray = new char* [numLines];
	for (int i = 0; i < numLines; i++) {
		charArray[i] = new char[v[i].size() + 1];
		std::strcpy(charArray[i], v[i].c_str());
	}

	std::cout << "starting parallelization" << '\n';

	loadParallelHashes("test", 4, charArray, numLines);


}


