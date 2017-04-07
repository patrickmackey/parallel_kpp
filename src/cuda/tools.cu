#include "tools.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

using namespace thrust;
using namespace std;

__global__ 
void newMinDist_kernel(const int n, const int m, const float *data, 
	float *minDists)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int indexOffset = index * m;
	const float *indexData = &data[indexOffset];
	float dist = 0.0f;
	for(int mi=0; mi<m; mi++)
	{
		float dx = indexData[mi] - const_seedData[mi];
		dist += dx*dx;
	}

	float prevDist = minDists[index];
	if(dist < prevDist)
		minDists[index] = dist;	
}

__global__ 
void newMinDist_kernel_OLD(const int n, const int m, const float *data, 
	const float *seedData, float *minDists)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int indexOffset = index * m;
	const float *indexData = &data[indexOffset];
	float dist = 0.0f;
	for(int mi=0; mi<m; mi++)
	{
		float dx = indexData[mi] - seedData[mi];
		dist += dx*dx;
	}

	float prevDist = minDists[index];
	if(dist < prevDist)
		minDists[index] = dist;	
}

__global__
void calcNorms_kernel(const int n, const int m, const float *data, float *norms)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int indexOffset = index * m;
	const float *indexData = &data[indexOffset];

	float dotProd = 0.0f;
	for(int mi=0; mi<m; mi++)
	{
		float val = indexData[mi];
		dotProd += val*val;
	}
	norms[index] = dotProd;
}


// Calculates the dot-product of a vector times itself
// (Used for the standard CPU version)
float calcNormSq(const float *vec, int length)
{
	float sum = 0.0f;	
	for(int i=0; i<length; i++)
	{
		sum += vec[i]*vec[i];
	}
	return sum;
}

// Calculates the dot product of two vectors
// (Used for the standard CPU version)
float calcDistSq(const float *vec1, const float *vec2, int length)
{
	float sum = 0.0f;
	for(int i=0; i<length; i++)
	{
		float dist = vec1[i]-vec2[i];
		sum += dist*dist;
	}
	return sum;
}

