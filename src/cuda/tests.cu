#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/generate.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"
#include "kkz.h"
#include "kpp.h"

using namespace thrust;
using namespace std;

/**
 * Functor struct for filling an array with random floating point values.
 */
struct RandFunctor
{
	float operator()()
	{
		return (float)rand()/RAND_MAX;
	}
};

void kkz_test_new(int n, int m, int k, int numTrials)
{
	float avgTime = 0.0f, minTime = 0.0f, maxTime = 0.0f;
	float avgTimeCopy = 0.0f, minTimeCopy = 0.0f, maxTimeCopy = 0.0f;

	//printf("=====================================\n");
	//printf("n = %d, m = %d, k = %d\n\n", n, m, k);

	for(int trial=0; trial<numTrials; trial++)
	{
		clock_t start, end;

		host_vector<int> h_membership(n);
		host_vector<float> h_data(n * m);

		// Fills our data with random values
		//thrust::generate(h_data.begin(), h_data.end(), RandFunctor());
		for(int i=0; i<n*m; i++)
			h_data[i] = (float)rand()/(float)RAND_MAX;

		host_vector<int> h_seeds(k);

		start = clock();
		double compTime = kkz(n, m, k, h_data, h_seeds);
		end = clock();
		/*if(k < 80)
		{
			printf("GPU KKZ seeds: ");
			for(int i=0; i<k; i++)
				printf("%d ", h_seeds[i]);
			printf("\n");
		}*/
		float gpuTime = (float)(end-start)/CLOCKS_PER_SEC;

		if(trial == 0)
		{
			minTime = compTime;
			maxTime = compTime;
			minTimeCopy = gpuTime;
			maxTimeCopy = gpuTime;
		}
		avgTime += compTime;
		avgTimeCopy += gpuTime;
		if(compTime < minTime) minTime = compTime;
		if(compTime > maxTime) maxTime = compTime;
		if(gpuTime < minTimeCopy) minTimeCopy = gpuTime;
		if(gpuTime > maxTimeCopy) maxTimeCopy = gpuTime;
	}
	avgTime /= numTrials;
	avgTimeCopy /= numTrials;
	//printf("(with copy): %f, %f, %f\n", minTimeCopy, avgTimeCopy, maxTimeCopy);
	//printf("(without copy): %f, %f, %f\n", minTime, avgTime, maxTime);

	cout << n << ", " << m << ", " << k << ", " << avgTime << ", ";
	cout << avgTime - minTime << ", " << maxTime - avgTime << ", ";
	cout << avgTimeCopy << ", " << avgTimeCopy - minTimeCopy << ", ";
	cout << maxTimeCopy - avgTimeCopy << endl;
}

void kpp_test_new(int n, int m, int k, int numTrials)
{
	float avgTime = 0.0f, minTime = 0.0f, maxTime = 0.0f;
	float avgTimeCopy = 0.0f, minTimeCopy = 0.0f, maxTimeCopy = 0.0f;

	//printf("=====================================\n");
	//printf("n = %d, m = %d, k = %d\n\n", n, m, k);

	for(int trial=0; trial<numTrials; trial++)
	{
		clock_t start, end;

		host_vector<int> h_membership(n);
		host_vector<float> h_data(n * m);

		// Fills our data with random values
		//thrust::generate(h_data.begin(), h_data.end(), RandFunctor());
		for(int i=0; i<n*m; i++)
			h_data[i] = (float)rand()/(float)RAND_MAX;

		host_vector<int> h_seeds(k);

		start = clock();
		double compTime = kpp(n, m, k, h_data, h_seeds);
		end = clock();
		/*if(k < 80)
		{
			printf("GPU KKZ seeds: ");
			for(int i=0; i<k; i++)
				printf("%d ", h_seeds[i]);
			printf("\n");
		}*/
		float gpuTime = (float)(end-start)/CLOCKS_PER_SEC;

		if(trial == 0)
		{
			minTime = compTime;
			maxTime = compTime;
			minTimeCopy = gpuTime;
			maxTimeCopy = gpuTime;
		}
		avgTime += compTime;
		avgTimeCopy += gpuTime;
		if(compTime < minTime) minTime = compTime;
		if(compTime > maxTime) maxTime = compTime;
		if(gpuTime < minTimeCopy) minTimeCopy = gpuTime;
		if(gpuTime > maxTimeCopy) maxTimeCopy = gpuTime;
	}
	avgTime /= numTrials;
	avgTimeCopy /= numTrials;
	//printf("(with copy): %f, %f, %f\n", minTimeCopy, avgTimeCopy, maxTimeCopy);
	//printf("(without copy): %f, %f, %f\n", minTime, avgTime, maxTime);

	cout << n << ", " << m << ", " << k << ", " << avgTime << ", ";
	cout << avgTime - minTime << ", " << maxTime - avgTime << ", ";
	cout << avgTimeCopy << ", " << avgTimeCopy - minTimeCopy << ", ";
	cout << maxTimeCopy - avgTimeCopy << endl;
}

void kkz_test(int n, int m, int k)
{
	clock_t start, end;

	host_vector<int> h_membership(n);
	host_vector<float> h_data(n * m);

	// Fills our data with random values
	//thrust::generate(h_data.begin(), h_data.end(), RandFunctor());
	for(int i=0; i<n*m; i++)
		h_data[i] = (float)rand()/(float)RAND_MAX;

	host_vector<int> h_seeds(k);

	printf("=====================================\n");
	printf("n = %d, m = %d, k = %d\n\n", n, m, k);

	start = clock();
	double compTime = kkz(n, m, k, h_data, h_seeds);
	end = clock();
	if(k < 80)
	{
		printf("GPU KKZ seeds: ");
		for(int i=0; i<k; i++)
			printf("%d ", h_seeds[i]);
		printf("\n");
	}
	float gpuTime = (float)(end-start)/CLOCKS_PER_SEC;
	printf("Time to compute (with copy): %f\n", gpuTime);
	printf("Time to compute (without copy): %f\n\n", compTime);

	start = clock();
	kkz_cpu(n, m, k, h_data, h_seeds);
	end = clock();
	if(k < 80)
	{
		printf("CPU KKZ seeds: ");
		for(int i=0; i<k; i++)
			printf("%d ", h_seeds[i]);
		printf("\n");
	}
	float cpuTime = (float)(end-start)/CLOCKS_PER_SEC;
	printf("Time to compute: %f\n", cpuTime);
	printf("\nSpeed up (with copy) = %fx\n", cpuTime/gpuTime);
	printf("Speed up (without copy) = %fx\n", cpuTime/compTime);
	printf("=====================================\n");
}

void kpp_test(int n, int m, int k)
{
	clock_t start, end;

	host_vector<int> h_membership(n);
	host_vector<float> h_data(n * m);

	// Fills our data with random values
	//thrust::generate(h_data.begin(), h_data.end(), RandFunctor());
	for(int i=0; i<n*m; i++)
		h_data[i] = (float)rand()/(float)RAND_MAX;

	host_vector<int> h_seeds(k);

	printf("=====================================\n");
	printf("n = %d, m = %d, k = %d\n\n", n, m, k);

	start = clock();
	double compTime = kpp(n, m, k, h_data, h_seeds);
	end = clock();
	if(k < 80)
	{
		printf("GPU k-means++ seeds: ");
		for(int i=0; i<k; i++)
			printf("%d ", h_seeds[i]);
		printf("\n");
	}
	float gpuTime = (float)(end-start)/CLOCKS_PER_SEC;
	printf("Time to compute (with copy): %f\n", gpuTime);
	printf("Time to compute (without copy): %f\n\n", compTime);

	start = clock();
	kpp_cpu(n, m, k, h_data, h_seeds);
	end = clock();
	if(k < 80)
	{
		printf("CPU k-means++ seeds: ");
		for(int i=0; i<k; i++)
			printf("%d ", h_seeds[i]);
		printf("\n");
	}
	float cpuTime = (float)(end-start)/CLOCKS_PER_SEC;
	printf("Time to compute: %f\n", cpuTime);
	printf("\nSpeed up (with copy) = %fx\n", cpuTime/gpuTime);
	printf("Speed up (without copy) = %fx\n", cpuTime/compTime);
	printf("=====================================\n");
}
