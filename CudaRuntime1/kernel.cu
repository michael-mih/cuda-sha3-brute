#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <openssl/sha.h>
#include <string.h>
#include "SHA-3/hex.h"
#include "SHA-3/Keccak.cuh"
#include "SHA-3/HashFunction.cuh"

//TODO convert sha3 etc to .cu 

__device__ std::string 
hex_encode(std::string inpt, KeccakBase& k) {
	const uint8_t* byte_array = reinterpret_cast<const uint8_t*>(inpt.data());
	k.addData(byte_array, 0, inpt.length());
	std::vector<unsigned char> op = k.digest();
	std::ostringstream b;
	for (auto& oi : op)
	{
		Hex(oi, [&b](unsigned char a) { b << a; });
	}
	return b.str();
}

__global__ void
bruteSearch(std::string* hash, std::string** wordlist, size_t n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x; //blockDim.x threads per block
	if (index < n) {
		printf("index %d", hex_encode(*wordlist[index], Sha3(256)));
	}
}


cudaError_t loadParallelHashes(std::string desiredHash, std::string* wordlist, size_t wordlistLength) //todo array decay?
{
#define THREADS_PER_BLOCK 512;
	std::string* dev_desiredHash = 0;
	std::string** dev_wordlist = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	//malloc and cpy for desired hash
	cudaStatus = cudaMalloc((void **)&dev_desiredHash, desiredHash.size());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_desiredHash, &desiredHash, desiredHash.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//malloc and cpy for wordlist
	cudaStatus = cudaMalloc((void ***)&dev_wordlist, wordlistLength);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_wordlist, &wordlist, wordlistLength, cudaMemcpyHostToDevice);

	int numBlocks = wordlistLength / THREADS_PER_BLOCK; 

	//bruteSearch<<<(wordlistLength/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dev_desiredHash, dev_wordlist);
	bruteSearch <<<1, 512 >> > (dev_desiredHash, dev_wordlist, wordlistLength);

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
	/*std::string raw("test");
	std::cout << hex_encode(raw, Sha3(256)); */

	std::string curLine;
	std::ifstream stream("file.txt");
	int numLines = 0;
	while (std::getline(stream, curLine)) {
		numLines++;
	}
	std::string* wordlist = new std::string[numLines];
	std::ifstream readStream("file.txt");
	int i = 0;
	while (std::getline(readStream, curLine)) {
		wordlist[i] = curLine;
		std::cout << wordlist[i] << '\n';
		i++;
	}

	/*int boundary = numLines / numThreads;
	int i = 0; int j = 0;

	while (std::getline(readStream, curLine)) {
		if (i > boundary) {
			i = 0;
			j++;
		}
		wordlistChunks[j] += curLine;
		i++;
	} */


	std::cout << "starting parallelization" << '\n';
	
	loadParallelHashes("test", wordlist, numLines);

	
}


	