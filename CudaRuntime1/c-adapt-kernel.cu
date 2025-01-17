#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "input.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <tuple>
#include "C-SHA-3/sha3.cuh"

#define SHA3_ASSERT( x )
#define SHA3_TRACE( format, ...)
#define SHA3_TRACE_BUF(format, buf, l)

/*
 * This flag is used to configure "pure" Keccak, as opposed to NIST SHA3.
 */
#define SHA3_USE_KECCAK_FLAG 0x80000000
#define SHA3_CW(x) ((x) & (~SHA3_USE_KECCAK_FLAG))


#if defined(_MSC_VER)
#define SHA3_CONST(x) x
#else
#define SHA3_CONST(x) x##L
#endif

#ifndef SHA3_ROTL64
#define SHA3_ROTL64(x, y) \
	(((x) << (y)) | ((x) >> ((sizeof(uint64_t)*8) - (y))))
#endif

__device__ static const uint64_t keccakf_rndc[24] = {
	SHA3_CONST(0x0000000000000001UL), SHA3_CONST(0x0000000000008082UL),
	SHA3_CONST(0x800000000000808aUL), SHA3_CONST(0x8000000080008000UL),
	SHA3_CONST(0x000000000000808bUL), SHA3_CONST(0x0000000080000001UL),
	SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008009UL),
	SHA3_CONST(0x000000000000008aUL), SHA3_CONST(0x0000000000000088UL),
	SHA3_CONST(0x0000000080008009UL), SHA3_CONST(0x000000008000000aUL),
	SHA3_CONST(0x000000008000808bUL), SHA3_CONST(0x800000000000008bUL),
	SHA3_CONST(0x8000000000008089UL), SHA3_CONST(0x8000000000008003UL),
	SHA3_CONST(0x8000000000008002UL), SHA3_CONST(0x8000000000000080UL),
	SHA3_CONST(0x000000000000800aUL), SHA3_CONST(0x800000008000000aUL),
	SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008080UL),
	SHA3_CONST(0x0000000080000001UL), SHA3_CONST(0x8000000080008008UL)
};

__device__ static const unsigned keccakf_rotc[24] = {
	1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62,
	18, 39, 61, 20, 44
};

__device__ static const unsigned keccakf_piln[24] = {
	10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20,
	14, 22, 9, 6, 1
};


__device__ static void
keccakf(uint64_t s[25])
{
	int i, j, round;
	uint64_t t, bc[5];
#define KECCAK_ROUNDS 24

	for (round = 0; round < KECCAK_ROUNDS; round++) {

		/* Theta */
		for (i = 0; i < 5; i++)
			bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];

		for (i = 0; i < 5; i++) {
			t = bc[(i + 4) % 5] ^ SHA3_ROTL64(bc[(i + 1) % 5], 1);
			for (j = 0; j < 25; j += 5)
				s[j + i] ^= t;
		}

		/* Rho Pi */
		t = s[1];
		for (i = 0; i < 24; i++) {
			j = keccakf_piln[i];
			bc[0] = s[j];
			s[j] = SHA3_ROTL64(t, keccakf_rotc[i]);
			t = bc[0];
		}

		/* Chi */
		for (j = 0; j < 25; j += 5) {
			for (i = 0; i < 5; i++)
				bc[i] = s[j + i];
			for (i = 0; i < 5; i++)
				s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
		}

		/* Iota */
		s[0] ^= keccakf_rndc[round];
	}
}

__device__ enum SHA3_FLAGS sha3_SetFlags(void* priv, enum SHA3_FLAGS);

__device__ enum SHA3_FLAGS
	sha3_SetFlags(void* priv, enum SHA3_FLAGS flags)
{
	sha3_context* ctx = (sha3_context*)priv;
	//flags &= SHA3_FLAGS_KECCAK;
	flags = static_cast<SHA3_FLAGS>(static_cast<int>(flags) & static_cast<int>(SHA3_FLAGS_KECCAK));
	ctx->capacityWords |= (flags == SHA3_FLAGS_KECCAK ? SHA3_USE_KECCAK_FLAG : 0);
	return flags;
}

__device__ sha3_return_t
sha3_Init(void* priv, unsigned bitSize) {
	sha3_context* ctx = (sha3_context*)priv;
	if (bitSize != 256 && bitSize != 384 && bitSize != 512)
		return SHA3_RETURN_BAD_PARAMS;
	memset(ctx, 0, sizeof(*ctx));
	ctx->capacityWords = 2 * bitSize / (8 * sizeof(uint64_t));
	return SHA3_RETURN_OK;
}

__device__ void
sha3_Update(void* priv, void const* bufIn, size_t len)
{
	sha3_context* ctx = (sha3_context*)priv;

	/* 0...7 -- how much is needed to have a word */
	unsigned old_tail = (8 - ctx->byteIndex) & 7;

	size_t words;
	unsigned tail;
	size_t i;

	const uint8_t* buf = static_cast<const uint8_t*>(bufIn);

	SHA3_TRACE_BUF("called to update with:", buf, len);

	SHA3_ASSERT(ctx->byteIndex < 8);
	SHA3_ASSERT(ctx->wordIndex < sizeof(ctx->u.s) / sizeof(ctx->u.s[0]));

	if (len < old_tail) {        /* have no complete word or haven't started
								 * the word yet */
		SHA3_TRACE("because %d<%d, store it and return", (unsigned)len,
			(unsigned)old_tail);
		/* endian-independent code follows: */
		while (len--)
			ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);
		SHA3_ASSERT(ctx->byteIndex < 8);
		return;
	}

	if (old_tail) {              /* will have one word to process */
		SHA3_TRACE("completing one word with %d bytes", (unsigned)old_tail);
		/* endian-independent code follows: */
		len -= old_tail;
		while (old_tail--)
			ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);

		/* now ready to add saved to the sponge */
		ctx->u.s[ctx->wordIndex] ^= ctx->saved;
		SHA3_ASSERT(ctx->byteIndex == 8);
		ctx->byteIndex = 0;
		ctx->saved = 0;
		if (++ctx->wordIndex ==
			(SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords))) {
			keccakf(ctx->u.s);
			ctx->wordIndex = 0;
		}
	}

	/* now work in full words directly from input */

	SHA3_ASSERT(ctx->byteIndex == 0);

	words = len / sizeof(uint64_t);
	tail = len - words * sizeof(uint64_t);

	SHA3_TRACE("have %d full words to process", (unsigned)words);

	for (i = 0; i < words; i++, buf += sizeof(uint64_t)) {
		const uint64_t t = (uint64_t)(buf[0]) |
			((uint64_t)(buf[1]) << 8 * 1) |
			((uint64_t)(buf[2]) << 8 * 2) |
			((uint64_t)(buf[3]) << 8 * 3) |
			((uint64_t)(buf[4]) << 8 * 4) |
			((uint64_t)(buf[5]) << 8 * 5) |
			((uint64_t)(buf[6]) << 8 * 6) |
			((uint64_t)(buf[7]) << 8 * 7);
#if defined(__x86_64__ ) || defined(__i386__)
		SHA3_ASSERT(memcmp(&t, buf, 8) == 0);
#endif
		ctx->u.s[ctx->wordIndex] ^= t;
		if (++ctx->wordIndex ==
			(SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords))) {
			keccakf(ctx->u.s);
			ctx->wordIndex = 0;
		}
	}

	SHA3_TRACE("have %d bytes left to process, save them", (unsigned)tail);

	/* finally, save the partial word */
	SHA3_ASSERT(ctx->byteIndex == 0 && tail < 8);
	while (tail--) {
		SHA3_TRACE("Store byte %02x '%c'", *buf, *buf);
		ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);
	}
	SHA3_ASSERT(ctx->byteIndex < 8);
	SHA3_TRACE("Have saved=0x%016" PRIx64 " at the end", ctx->saved);
}

__device__ void const*
sha3_Finalize(void* priv)
{
	sha3_context* ctx = (sha3_context*)priv;

	SHA3_TRACE("called with %d bytes in the buffer", ctx->byteIndex);

	/* Append 2-bit suffix 01, per SHA-3 spec. Instead of 1 for padding we
	 * use 1<<2 below. The 0x02 below corresponds to the suffix 01.
	 * Overall, we feed 0, then 1, and finally 1 to start padding. Without
	 * M || 01, we would simply use 1 to start padding. */

	uint64_t t;

	if (ctx->capacityWords & SHA3_USE_KECCAK_FLAG) {
		/* Keccak version */
		t = (uint64_t)(((uint64_t)1) << (ctx->byteIndex * 8));
	}
	else {
		/* SHA3 version */
		t = (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << ((ctx->byteIndex) * 8));
	}

	ctx->u.s[ctx->wordIndex] ^= ctx->saved ^ t;

	ctx->u.s[SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords) - 1] ^=
		SHA3_CONST(0x8000000000000000UL);
	keccakf(ctx->u.s);

	/* Return first bytes of the ctx->s. This conversion is not needed for
	 * little-endian platforms e.g. wrap with #if !defined(__BYTE_ORDER__)
	 * || !defined(__ORDER_LITTLE_ENDIAN__) || __BYTE_ORDER__!=__ORDER_LITTLE_ENDIAN__
	 *    ... the conversion below ...
	 * #endif */
	{
		unsigned i;
		for (i = 0; i < SHA3_KECCAK_SPONGE_WORDS; i++) {
			const unsigned t1 = (uint32_t)ctx->u.s[i];
			const unsigned t2 = (uint32_t)((ctx->u.s[i] >> 16) >> 16);
			ctx->u.sb[i * 8 + 0] = (uint8_t)(t1);
			ctx->u.sb[i * 8 + 1] = (uint8_t)(t1 >> 8);
			ctx->u.sb[i * 8 + 2] = (uint8_t)(t1 >> 16);
			ctx->u.sb[i * 8 + 3] = (uint8_t)(t1 >> 24);
			ctx->u.sb[i * 8 + 4] = (uint8_t)(t2);
			ctx->u.sb[i * 8 + 5] = (uint8_t)(t2 >> 8);
			ctx->u.sb[i * 8 + 6] = (uint8_t)(t2 >> 16);
			ctx->u.sb[i * 8 + 7] = (uint8_t)(t2 >> 24);
		}
	}

	SHA3_TRACE_BUF("Hash: (first 32 bytes)", ctx->u.sb, 256 / 8);

	return (ctx->u.sb);
}

__device__ sha3_return_t
sha3_HashBuffer(unsigned bitSize, enum SHA3_FLAGS flags, const void* in, unsigned inBytes, void* out, unsigned outBytes) {
	sha3_return_t err;
	sha3_context c;

	err = sha3_Init(&c, bitSize);
	if (err != SHA3_RETURN_OK)
		return err;
	if (sha3_SetFlags(&c, flags) != flags) {
		return SHA3_RETURN_BAD_PARAMS;
	}
	sha3_Update(&c, in, inBytes);
	const void* h = sha3_Finalize(&c);

	if (outBytes > bitSize / 8)
		outBytes = bitSize / 8;
	memcpy(out, h, outBytes);
	return SHA3_RETURN_OK;
}

//microsoft crt memcmp implementation
__device__ int 
own_memcmp(const void* buf1, const void* buf2, size_t count)
{
	if (!count)
		return(0);

	while (--count && *(char*)buf1 == *(char*)buf2) {
		buf1 = (char*)buf1 + 1;
		buf2 = (char*)buf2 + 1;
	}

	return(*((unsigned char*)buf1) - *((unsigned char*)buf2));
}

static int blocksPerGrid;
static int threadsPerBlock;

__global__ void
bruteSearch(char* hash, char* wordlist, size_t n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x; //blockDim.x threads per block
	if (index < n-1 && wordlist[index] == '\0') {

		sha3_context c;
		int size = 1;
		while (*(wordlist + index + size) != '\0') {
			size++;
		}

		unsigned char buf[32];
		sha3_HashBuffer(256, SHA3_FLAGS_NONE, wordlist + index + 1, size-1, buf, sizeof(buf));
		if (own_memcmp(hash, buf, 32) == 0) {
			printf("HASH BROKEN WITH WORD \"%s\"", wordlist + index + 1);
		}

	}
}


cudaError_t loadParallelHashes(unsigned char* desiredHash, size_t unhashed_size, const char* wordlist, size_t wordlist_size) //todo array decay?
{
#define THREADS_PER_BLOCK 512;
	char* dev_desiredHash = 0;
	char* dev_wordlist = 0;

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
	cudaStatus = cudaMemcpy(dev_desiredHash, desiredHash, unhashed_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//malloc and cpy for wordlist
	cudaStatus = cudaMalloc((void**)&dev_wordlist, wordlist_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_wordlist, wordlist, wordlist_size, cudaMemcpyHostToDevice);



	bruteSearch << <blocksPerGrid, threadsPerBlock >> > (dev_desiredHash, dev_wordlist, wordlist_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch. (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();


Error:
	return cudaStatus;
}


int main(int argc, char* argv[]) {
	int numThreads;
	if (argc != 3) {
		std::cerr << "arg1 hash arg2 wordlist";
		return 1;
	}

	threadsPerBlock = 256;

	unsigned char buf[32];
	hexInputToBytes(argv[1], buf, sizeof(buf));

	std::string charArray = "";
	charArray += '\0';
	size_t size = 0;
	int workload = 0;
	
	std::cout << "loading wordlist" << "\n";
	std::tuple<size_t, size_t> t = readFile(argv[2], charArray);
	
	blocksPerGrid = (std::ceil(std::get<0>(t) / threadsPerBlock)); //ceil, need at least one block

	std::cout << "starting parallelization" << '\n';
	loadParallelHashes(buf, 32, charArray.c_str(), std::get<1>(t));
	std::cout << '\n' << "finished!";
	return 0;
}


