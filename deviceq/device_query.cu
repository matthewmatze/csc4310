/*
	Name: Matthew Matze
	Date: 9/28/2016
	Class: csc4310
	Location: ~/csc3210/deviceq

	General Summary of Program

	The program is set up to show the various device properties to the screen
	
	To Compile:

	nvcc device_query.cu -o device_query
	
	To Execute:
	
	device_query

*/

#include<stdio.h>

void printDevProp(cudaDeviceProp devProp);
/*

	The function prints out some of the properties of the cudaDeviceProp struct
	
	Parameters:The struct to with which the device info shall be taken from

	Postcondition: The specified parameters have been outputed to the screen

*/

int main(void){

	int devCnt;
	cudaGetDeviceCount(&devCnt);
	for(int i=0;i<devCnt;i++){
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp,i);
		printDevProp(devProp);
	}
}

void printDevProp(cudaDeviceProp devProp){

	printf("Device name: %s\n", devProp.name);
	printf("Major: %d\n",devProp.major);
	printf("Minor: %d\n",devProp.minor);
	printf("TotalGlobalMem(Bytes): %d\n",devProp.totalGlobalMem);
	printf("SharedMemPerBlock: %d\n",devProp.sharedMemPerBlock);
	printf("RegsPerBlock: %d\n",devProp.regsPerBlock);
	printf("WarpSize: %d\n",devProp.warpSize);
	printf("MemPitch: %d\n",devProp.memPitch);
	printf("MaxThreadsPerBlock: %d\n",devProp.maxThreadsPerBlock);
	printf("MaxThreadsPerMultiProcessor: %d\n",devProp.maxThreadsPerMultiProcessor);
	for(int i=0;i<3;i++){
		printf("MaxThreadsDim[%d]: %d\n", i ,devProp.maxThreadsDim[i]);
		printf("MaxGridSize[%d]: %d\n", i, devProp.maxGridSize[i]);
	}
	printf("ClockRate: %d\n",devProp.clockRate);
	printf("TotalConstMem: %d\n",devProp.totalConstMem);
	printf("TextureAlignment: %d\n",devProp.textureAlignment);
	printf("DeviceOverlap: %d\n",devProp.deviceOverlap);
	printf("MultiProcessorCount: %d\n",devProp.multiProcessorCount);
	printf("KernelExecTimeoutEnabled: %d\n",devProp.kernelExecTimeoutEnabled);
}
