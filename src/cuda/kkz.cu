#include "kkz.h"
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

// Uses CUDA instead of thrust for everything except reductions
/*double kkz_cuda(const int n, const int m, const int k, const host_vector<float> &h_data, 
		 host_vector<int> &h_outSeeds)
{
	// Copy data to device
	device_vector<float> d_data = h_data; // O(n*m)

	clock_t start = clock();
	
	// Stores shortest distance of data points to any existing seed
	device_vector<float> d_minDists(n);
	// Initialize all to "infinity"
	thrust::fill(d_minDists.begin(), d_minDists.end(), FLT_MAX);

	// --- Determine size of GPU thread blocks and grid ---	
	// 512 is largest size possible (as I understand it)
	int block_size = 512; 
	// Should give us the proper number, even if we go over
	int n_blocks = n / block_size + (n % block_size == 0 ? 0:1);
	// Define actual proper sizes
	dim3 dimGrid(n_blocks, 1, 1);
	dim3 dimBlock(block_size, 1, 1);

	// Cast our device collections to raw pointers
	float *raw_data = thrust::raw_pointer_cast(&d_data[0]);
	float *raw_minDists = thrust::raw_pointer_cast(&d_minDists[0]);

	// A list of indexes for our data, used for our functors
	// (0,1,2,3...n-1)
	device_vector<int> d_indices(n);
	thrust::sequence(d_indices.begin(), d_indices.end()); // O(n)

	// Calculate norm of data points
	device_vector<float> d_norms(n);
	float *raw_norms = thrust::raw_pointer_cast(&d_norms[0]);
	calcNorms_kernel<<<dimGrid,dimBlock>>>(n, m, raw_data, raw_norms);	
	//thrust::transform(d_indices.begin(), d_indices.end(), d_norms.begin(),
		//	CalcNormFunctor(n, m, k, d_data)); // O(n*m)

	// Find data point with highest norm
	device_vector<float>::iterator maxItr = thrust::max_element(d_norms.begin(), d_norms.end()); // O(n*log n)
	int maxIndex = maxItr - d_norms.begin();
	float max_val = *maxItr;
	//printf("Largest norm: %d = %f\n", maxIndex, max_val);
	h_outSeeds[0] = maxIndex;

	device_vector<float> d_seedData(m);

	// Find the rest of the seeds
	for(int ki=1; ki<k; ki++)
	{
		// Look at last seed
		int lastSeed = h_outSeeds[ki-1];		
		printf("lastSeed = %d\n", lastSeed);

		cudaMemcpyToSymbol(

		// Copy seed to constant device memory
		for(int mi=0; mi<m; mi++)
		{
			//const_seedData[mi] = d_data[lastSeed*m+mi];
			float val = const_seedData[mi];
			printf("%f ", val);
		}		
		printf("\n");
		for(int mi=0; mi<m; mi++)
		{
			float val = d_data[lastSeed*m+mi];
			printf("%f ", val);

		}
		printf("\n");
		//thrust::copy(d_data.begin() + lastSeed*m, d_data.begin() + (lastSeed+1)*m, d_seedData.begin());
		//float *rawSeedData = thrust::raw_pointer_cast(&d_seedData[0]);
		//for(int mi=0; mi<m; mi++)
		//{
			//float val = d_data[lastSeed*m+mi];
			//printf("%f ", val);
		//}
		//printf("\n");

		// Calculate distance of all other data points to last seed (if smaller
		// than previous distance)
		newMinDist_kernel<<<dimGrid,dimBlock>>>(n, m, raw_data, raw_minDists);			
		//newMinDist_kernel_OLD<<<dimGrid,dimBlock>>>(n, m, raw_data, rawSeedData, raw_minDists);			
		//newMinDist_kernel<<<dimGrid,dimBlock>>>(n, m, raw_data, lastSeed, raw_minDists);			

		// Pick seed with maximum smallest distance
		maxItr = thrust::max_element(d_minDists.begin(), d_minDists.end()); // O(n*log n)
		maxIndex = maxItr - d_minDists.begin();
		h_outSeeds[ki] = maxIndex;
	}

	clock_t end = clock();
	return (double)(end-start)/CLOCKS_PER_SEC;
}*/

// Our GPU based version of KKZ
double kkz(const int n, const int m, const int k, const host_vector<float> &h_data, 
		 host_vector<int> &h_outSeeds)
{
	// Copy data to device
	device_vector<float> d_data = h_data; // O(n*m)

	clock_t start = clock();

	// A list of indexes for our data, used for our functors
	// (0,1,2,3...n-1)
	device_vector<int> d_indices(n);
	thrust::sequence(d_indices.begin(), d_indices.end()); // O(n)

	// Calculate norm of data points
	device_vector<float> d_norms(n);
	thrust::transform(d_indices.begin(), d_indices.end(), d_norms.begin(),
			CalcNormFunctor(n, m, k, d_data)); // O(n*m)

	// Find data point with highest norm
	device_vector<float>::iterator maxItr = thrust::max_element(d_norms.begin(), d_norms.end()); // O(n*log n)
	int maxIndex = maxItr - d_norms.begin();
	float max_val = *maxItr;
	//printf("Largest norm: %d = %f\n", maxIndex, max_val);
	h_outSeeds[0] = maxIndex;

	// Stores shortest distance of data points to any existing seed
	device_vector<float> d_minDists(n);
	// Initialize all to "infinity"
	thrust::fill(d_minDists.begin(), d_minDists.end(), FLT_MAX);

	// Find the rest of the seeds
	for(int ki=1; ki<k; ki++)
	{
		// Calculate distance of all other data points to newest seed (if smaller
		// than previous distance)
		thrust::transform(d_indices.begin(), d_indices.end(), d_minDists.begin(), 
			CalcNewMinDist(n, m, k, d_data, d_minDists, h_outSeeds[ki-1])); // O(n*m)

		// Pick seed with maximum smallest distance
		maxItr = thrust::max_element(d_minDists.begin(), d_minDists.end()); // O(n*log n)
		maxIndex = maxItr - d_minDists.begin();
		h_outSeeds[ki] = maxIndex;
	}

	clock_t end = clock();
	return (double)(end-start)/CLOCKS_PER_SEC;
}

// CPU based, non-parallelized version of KKZ using the approach
// I normally would in standard C++
// (This runs much, much slower than the GPU version)
void kkz_cpu(const int n, const int m, const int k, const host_vector<float> &h_data, 
		 host_vector<int> &h_outSeeds)
{
	// Find data point with highest norm
	float maxNorm = -1.0f;
	int maxNormIndex = -1;
	for(int ni=0; ni<n; ni++)
	{
		float norm = calcNormSq(&h_data[ni*m], m);
		if(norm > maxNorm)
		{
			maxNorm = norm;
			maxNormIndex = ni;
		}
	}
	// Store first seed
	h_outSeeds[0] = maxNormIndex;

	// Stores shortest distance of data points to any existing seed
	host_vector<float> h_minDists(n);
	// Initialize all min dists to "infinity"
	thrust::fill(h_minDists.begin(), h_minDists.end(), FLT_MAX);

	// Find the rest of the seeds
	for(int ki=1; ki<k; ki++)
	{
		int prevSeed = h_outSeeds[ki-1];
		// Calculate distance of all other data points to newest seed (if smaller
		// than previous distance)
		for(int ni=0; ni<n; ni++)
		{
			float dist = calcDistSq(&h_data[prevSeed*m], &h_data[ni*m], m);
			if(dist < h_minDists[ni])
				h_minDists[ni] = dist;
		}

		// Pick seed with maximum smallest distance
		float maxDist = FLT_MIN;
		int maxDistIndex = -1;
		for(int ni=0; ni<n; ni++)
		{
			if(h_minDists[ni] > maxDist)
			{
				maxDist = h_minDists[ni];
				maxDistIndex = ni;
			}
		}
		h_outSeeds[ki] = maxDistIndex;
	}
}

// Non-parallel CPU version of KKZ using the same Thrust functions
// as the GPU version, but presumably running on the CPU from
// host memory.  Runs with similar performance to the GPU version.
// (Much, much faster than my other CPU version)
void kkz_cpu2(const int n, const int m, const int k, const host_vector<float> &h_data, 
		host_vector<int> &h_outSeeds)
{
	// A list of indexes for our data, used for our functors
	// (0,1,2,3...n-1)
	host_vector<int> h_indices(n);
	thrust::sequence(h_indices.begin(), h_indices.end());	

	// Calculate norm of data points
	host_vector<float> h_norms(n);
	thrust::transform(h_indices.begin(), h_indices.end(), h_norms.begin(),
			CalcNormFunctor(n, m, k, h_data));

	// Find data point with highest norm
	host_vector<float>::iterator maxItr = thrust::max_element(h_norms.begin(), h_norms.end());
	int maxIndex = maxItr - h_norms.begin();
	float max_val = *maxItr;
	//printf("Largest norm: %d = %f\n", maxIndex, max_val);
	h_outSeeds[0] = maxIndex;

	// Stores shortest distance of data points to any existing seed
	host_vector<float> h_minDists(n);
	// Initialize all to "infinity"
	thrust::fill(h_minDists.begin(), h_minDists.end(), FLT_MAX);

	// Find the rest of the seeds
	for(int ki=1; ki<k; ki++)
	{
		// Calculate distance of all other data points to newest seed (if smaller
		// than previous distance)
		thrust::transform(h_indices.begin(), h_indices.end(), h_minDists.begin(), 
			CalcNewMinDist(n, m, k, h_data, h_minDists, h_outSeeds[ki-1]));

		// Pick seed with maximum smallest distance
		maxItr = thrust::max_element(h_minDists.begin(), h_minDists.end());
		maxIndex = maxItr - h_minDists.begin();
		h_outSeeds[ki] = maxIndex;
	}
}
