
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

using namespace thrust;
using namespace std;

#define NEAR_ZERO .00001

struct FindMeansFunctor
{
	int _n, _m, _k;
	const float *_d_data, *_d_means;

	FindMeansFunctor(const int n, const int m, const int k, 
		const device_vector<float> &d_data, const device_vector<float> &d_means)
	{
		_d_data = thrust::raw_pointer_cast(&d_data[0]);
		_d_means = thrust::raw_pointer_cast(&d_means[0]);
		_n = n;
		_m = m;
		_k = k;
	}

	__host__ __device__
	int operator()(int index)
	{		
		int indexOffset = index*_m;
		int closestMean = -1;
		float closestDist = FLT_MAX;
		// Find closest mean to this element
		for(int ki=0; ki<_k; ki++)
		{
			float dotProd = 0.0f;
			int meanOffset = ki * _m;
			for(int mi=0; mi<_m; mi++)
			{
				dotProd += _d_data[indexOffset+mi] * _d_means[meanOffset+mi];
			}
			if(dotProd < closestDist)
			{
				closestDist = dotProd;
				closestMean = ki;
			}
		}
		return closestMean;
	}
};

bool calc_means(const int n, const int m, const int k, 
				const host_vector<float> &h_data, const host_vector<int> &h_membership, 
				const host_vector<float> &h_old_means, host_vector<float> &h_new_means, 
				host_vector<int> &h_cluster_sizes)
{
	// Set all new means initially to 0 and cluster sizes to 0
	for(int ki=0; ki<k; ki++)
	{
		h_cluster_sizes[ki] = 0;
		int offset = ki*m;
		for(int mi=0; mi<m; mi++)
		{
			h_new_means[offset + mi] = 0.0f;
		}
	}

	// Sum up
	for(int ni=0; ni<n; ni++)
	{
		int dataOffset = ni*m;
		int ki = h_membership[ni];
		int meanOffset = ki*m;
		for(int mi=0; mi<m; mi++)
		{
			h_new_means[meanOffset+mi] += h_data[dataOffset+mi];
		}
		h_cluster_sizes[ki]++;
	}

	// Find average
	for(int ki=0; ki<k; ki++)
	{
		int meanOffset = ki*m;
		float div = 1.0f / h_cluster_sizes[ki];
		for(int mi=0; mi<m; mi++)
		{
			h_new_means[meanOffset+mi] *= div;
		}
	}

	// Find if it's converged or not
	bool converged = true;
	int length = m * k;
	for(int i=0; i<length; i++)
	{
		if(abs(h_new_means[i] - h_old_means[i]) > NEAR_ZERO)
		{
			converged = false;
			break;
		}
	}
	return converged;
}

void kmeans_thrust(const int n, const int m, const int k, 
	const host_vector<float> &h_data, const host_vector<int> &h_seeds, 
	host_vector<int> &h_membership)
{		
	// Copying data to device
	device_vector<float> d_data = h_data;

	// Creating membership array
	device_vector<int> d_membership(n);

	// Creating initial means
	device_vector<float> d_means(k*m);
	host_vector<float> h_old_means(k*m);
	for(int ki=0; ki<k; ki++)
	{
		int seed = h_seeds[ki];		
		int start = seed * m;
		int end = start + m;
		// Copy data to mean
		thrust::copy(d_data.begin() + start, d_data.begin() + end, d_means.begin() + ki * m);
		thrust::copy(h_data.begin() + start, h_data.begin() + end, h_old_means.begin() + ki * m);		
	}
	// "new means" blank for now (used to test for convergence)
	host_vector<float> h_new_means(k*m);

	// stores cluster sizes
	host_vector<int> h_cluster_sizes(k);

	// A list of indexes for our data, used for our k-means functor
	// (0,1,2,3...n-1)
	device_vector<int> d_indices(n);
	thrust::sequence(d_indices.begin(), d_indices.end());	

	// Keep track of the number of iterations required to solve it
	int numIters = 0;
	// Flags when we've converged on a solution
	bool converged = false;
	// Loop until converged
	while(!converged)
	{
		numIters++;

		printf("Calling thrust transform...\n");

		// Find closest mean for each data point (running on GPU)
		// and assigns that mean's index to the element's membership
		thrust::transform(d_indices.begin(), d_indices.end(), d_membership.begin(),
			FindMeansFunctor(n, m, k, d_data, d_means));

		printf("done!\n");

		printf("Copying membership back to host...\n");		

		// Copy membership to host (for calculating means)
		//h_membership = d_membership;
		thrust::copy(d_membership.begin(), d_membership.end(), h_membership.begin());

		printf("done!\n");
		printf("membership: ");
		for(int i=0; i<n; i++)
			printf("%d ", h_membership[i]);
		printf("\n");

		printf("Calc new means...\n");

		// Calculates new means on CPU serially (for now at least)
		// and copies over old means with the new means. Returns 
		// if we've converged or not.
		converged = calc_means(n, m, k, h_data, h_membership, h_old_means, 
			h_new_means, h_cluster_sizes);

		printf("done!\n");

		if(!converged)
		{
			// Copy new means to the device and to old means
			//d_means = h_new_means;
			//h_old_means = h_new_means;
			thrust::copy(h_new_means.begin(), h_new_means.end(), d_means.begin());
			thrust::copy(h_new_means.begin(), h_new_means.end(), h_old_means.begin());
		}

		char blank[100];
		scanf("%s", blank);
	}

	printf("Number of iterations = %d\n", numIters);
	printf("Cluster sizes: ");
	for(int ki=0; ki<k; ki++)
		printf("%d ", h_cluster_sizes[ki]);
	printf("\n");
}

