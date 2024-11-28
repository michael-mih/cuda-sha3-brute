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

int main() {
	std::string raw("test");
	std::cout << hex_encode(raw, Sha3(256));
}


cudaError_t loadParallelHashes(std::string desiredHash, std::string wordlist, int numThreads)
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




	//malloc for chunk of wordlist
	for (int i = 0; i < numThreads; i++) {
		std::string* a = new std::string();
		//todo track devptrs and divide file input 
		cudaStatus = cudaMalloc((void **)&a, desiredHash.size() / numThreads);
	}
	

Error:
	return cudaStatus;
}

	