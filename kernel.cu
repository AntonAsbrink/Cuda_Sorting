#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>


__global__ void oddEvenGPU(int *arr, int arrSize)
{
    int i = (blockIdx.x*blockDim.x + threadIdx.x) * 2;

	if (i < arrSize) {
		for (int j = 0; j < arrSize/2; j++) {
			for (int k = i; k < arrSize; k += blockDim.x) {
				if (k + 1 < arrSize) {
					if (arr[k] > arr[k + 1]) {
						int temp = arr[k];
						arr[k] = arr[k + 1];
						arr[k + 1] = temp;
					}
				}
			}
			__syncthreads();
			i++;
			for (int k = i; k < arrSize; k += blockDim.x) {
				if (k + 1 < arrSize) {
					if (arr[k] > arr[k + 1]) {
						int temp = arr[k];
						arr[k] = arr[k + 1];
						arr[k + 1] = temp;
					}
				}
			}
			__syncthreads();
			i--;
		}
	}
}

__global__ void oddEvenBlockGPU(int *arr, int arrSize, int oddEven)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x) * 2 + oddEven; //oddEven should be a 0 or a 1 depending if its running even or odd phase

	int gridSize = gridDim.x * gridDim.y * gridDim.z;

	if (i < arrSize) {
		for (int k = i; k < arrSize; k += gridSize * blockDim.x) {
			if (k + 1 < arrSize) {
				if (arr[k] > arr[k + 1]) {
					int temp = arr[k];
					arr[k] = arr[k + 1];
					arr[k + 1] = temp;
				}
			}
		}
	}
}

__host__
void oddEvenCPU(int * arr, int arrSize) {
	for (int i = 0; i < arrSize; i++) {
		for (int j = i % 2; j < arrSize; j+=2) {
			if (j + 1 < arrSize) {

				if (arr[j] > arr[j + 1]) {
					int temp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = temp;
				}
			}
		}
	}
}

__host__
bool isSorted(int *arr, int arrSize) {
	for (int i = 0; i < arrSize-1; i++) {
		if (arr[i] > arr[i + 1]) {
			return false;
		}
	}
	return true;
}

__host__
int main(int argc, char *argv[])
{
	//Generate array values
	srand(time(NULL));

	int arrSize = std::stoi(std::string(argv[1]));

	int *arrCPU = new int[arrSize];
	int *arrGPU;
	for (int i = 0; i < arrSize; i++) {
		arrCPU[i] = rand() % 1000;
	}


	//CPU
	if (std::strcmp(argv[2], "cpu") == 0) {
		oddEvenCPU(arrCPU, arrSize);

		if (std::strcmp(argv[5], "output") == 0) {
			std::ofstream file;
			file.open("output.txt");
			file << "CPU is sorted?: " << (isSorted(arrCPU, arrSize) ? "True" : "False") << std::endl;
			for (int i = 0; i < arrSize; i++) {
				file << arrCPU[i] << " ";
			}
		}
	}

	//GPU pararellism test
	if (std::strcmp(argv[2], "gpu") == 0) {
		//CUDA init
		cudaError_t cudasStatus = cudaSetDevice(1);
		if (cudaMalloc(&arrGPU, arrSize * sizeof(int)) == cudaErrorMemoryAllocation)
			std::cout << "CUDA memory allow failed" << std::endl;
		cudaMemcpy(arrGPU, arrCPU, arrSize * sizeof(int), cudaMemcpyHostToDevice);

		if (std::strcmp(argv[4], "0") == 0) {
			oddEvenGPU << <1, std::stoi(std::string(argv[3])) >> > (arrGPU, arrSize);
		}
		else {
			for (int i = 0; i < arrSize; i++) {
				oddEvenBlockGPU << < std::stoi(std::string(argv[4])), std::stoi(std::string(argv[3])) >> > (arrGPU, arrSize, i % 2);
			}
		}

		int *g = new int[arrSize];
		cudaMemcpy(g, arrGPU, arrSize * sizeof(int), cudaMemcpyDeviceToHost);

		if (std::strcmp(argv[5], "output") == 0) {
			std::ofstream file;
			file.open("output.txt");
			file << "GPU is sorted?: " << (isSorted(g, arrSize) ? "True" : "False") << std::endl;
			for (int i = 0; i < arrSize; i++) {
				file << g[i] << " ";
			}
			delete(g);
		}
	}
	

	cudaFree(arrGPU);
	delete(arrCPU);

    return 0;
}