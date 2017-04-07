#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace thrust;
using namespace std;

// Constant GPU memory for holding the newest seed when
// calculating the new minimum distances
__constant__ float const_seedData[1024];

__global__ 
void newMinDist_kernel(const int n, const int m, const float *data, float *minDists);

__global__ 
void newMinDist_kernel_OLD(const int n, const int m, const float *data, 
	const float *seedData, float *minDists);

__global__
void calcNorms_kernel(const int n, const int m, const float *data, float *norms);

// Calculates the norm of a vector from our matrix of data
// at the given index
struct CalcNormFunctor
{
	int _n, _m, _k;
	const float *_d_data;

	// Stores our data so the function can work from just an index
	// (Using device memory)
	CalcNormFunctor(const int n, const int m, const int k, 
		const device_vector<float> &d_data)
	{
		_d_data = thrust::raw_pointer_cast(&d_data[0]);
		_n = n;
		_m = m;
		_k = k;
	}

	// Stores our data so the function can work from just an index
	// (Using host memory)
	CalcNormFunctor(const int n, const int m, const int k, 
		const host_vector<float> &h_data)
	{
		_d_data = thrust::raw_pointer_cast(&h_data[0]);
		_n = n;
		_m = m;
		_k = k;
	}

	// Computes the norm of the vector at the given index
	__host__ __device__
	float operator()(int index)
	{		
		int indexOffset = index*_m;
		const float *seedData = &_d_data[indexOffset];
		float dotProd = 0.0f;
		for(int mi=0; mi<_m; mi++)
		{
			float val = seedData[mi];
			dotProd += val * val;
		}
		return dotProd;
	}
};

// Calculates the new minimum distance for a point from
// all the previously selected seeds
struct CalcNewMinDist
{
	int _n, _m, _k;
	const float *_d_data;
	const float *_d_minDists;
	int _seed;

	// Stores the data and current min dists as well as the index
	// of the new seed (using device memory)
	CalcNewMinDist(const int n, const int m, const int k, 
		const device_vector<float> &d_data, const device_vector<float> &d_minDists, int seed)
	{
		_d_data = thrust::raw_pointer_cast(&d_data[0]);
		_d_minDists = thrust::raw_pointer_cast(&d_minDists[0]);
		_n = n;
		_m = m;
		_k = k;
		_seed = seed;
	}

	// Stores the data and current min dists as well as the index
	// of the new seed (using host memory)
	CalcNewMinDist(const int n, const int m, const int k, 
		const host_vector<float> &h_data, const host_vector<float> &h_minDists, int seed)
	{
		_d_data = thrust::raw_pointer_cast(&h_data[0]);
		_d_minDists = thrust::raw_pointer_cast(&h_minDists[0]);
		_n = n;
		_m = m;
		_k = k;
		_seed = seed;
	}

	// Calculates the distance of the data point at index from the
	// newest seed and sets the distance as the new min distance if
	// it is smaller than the previous one
	__host__ __device__
	float operator()(int index)
	{		
		int indexOffset = index*_m;
		int seedOffset = _seed*_m;
		const float *indexData = &_d_data[indexOffset];
		const float *seedData = &_d_data[seedOffset];		
		float dist = 0.0f;
		for(int mi=0; mi<_m; mi++)
		{
			float dx = indexData[mi] - seedData[mi];
			dist += dx*dx;
		}
		/*float dx = indexData[0] - seedData[0];		
		dist += dx*dx;
		dx = indexData[1] - seedData[1];
		dist += dx*dx;
		dx = indexData[2] - seedData[2];
		dist += dx*dx;
		dx = indexData[3] - seedData[3];
		dist += dx*dx;*/
		
		float prevDist = _d_minDists[index];
		if(dist < prevDist)
			return dist;
		else
			return prevDist;
	}
};

// Calculates the dot-product of a vector times itself
// (Used for the standard CPU version)
float calcNormSq(const float *vec, int length);

// Calculates the dot product of two vectors
// (Used for the standard CPU version)
float calcDistSq(const float *vec1, const float *vec2, int length);

