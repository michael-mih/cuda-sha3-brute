#include <iostream>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <openssl/sha.h>
#include <string.h>

#include "SHA-3/hex.h"
#include "SHA-3/Keccak.h"
#include "SHA-3/HashFunction.h"

std::string hex_encode(std::string inpt, KeccakBase& k) {
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

int main(int argc, char *argv[]) {
	int numThreads;
	if (argc == 2) {
		numThreads = atoi(argv[1]);
	}
	else {
		numThreads = 2;
	}
	std::string raw("test");
	std::cout << hex_encode(raw, Sha3(256));

	std::string curLine;
	std::ifstream stream("file.txt");
	int numLines = 0;
	while (std::getline(stream, curLine)) {
		numLines++;
	}
	std::string* wordlistChunks = new std::string[numThreads];
	std::ifstream readStream("file.txt");
	
	int boundary = numLines / numThreads;
	int i = 0; int j = 0;
	while (std::getline(readStream, curLine)) {
		if (i < boundary) {
			i = 0;
			j++;
		}
		wordlistChunks[j] = curLine;
		i++;
	}

}


cudaError_t loadParallelHashes(std::string desiredHash, std::string* wordlistChunks) //todo array decay?
{
	std::string* dev_desiredHash = 0;


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	//malloc for desired hash
	cudaStatus = cudaMalloc((void **)&dev_desiredHash, desiredHash.size());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	std::string** devPtrArr = new std::string * [wordlistChunks->size()]; //todo, pointer logic?
	//malloc for chunk of wordlist
	for (int i = 0; i < wordlistChunks->size(); i++) {
		devPtrArr[i] = new std::string();
		//todo track devptrs and divide file input 
		cudaStatus = cudaMalloc((void **)&devPtrArr[i], desiredHash.size() / wordlistChunks->size());
	}

	cudaStatus = cudaMemcpy(dev_desiredHash, &desiredHash, desiredHash.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for (int i = 0; i < numThreads; i++) {
		cudaStatus = cudaMemcpy(devPtrArr[i], )
	}


Error:
	return cudaStatus;
}

	