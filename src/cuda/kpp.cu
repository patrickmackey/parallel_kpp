#include "kpp.h"
#include "tools.h"
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <iostream>
#include <time.h>

using namespace std;

typedef tuple<int,float> ProbItem;
typedef device_vector<int>::iterator IntIterator;
typedef device_vector<float>::iterator FloatIterator;
typedef tuple<IntIterator, FloatIterator> IteratorTuple;
typedef zip_iterator<IteratorTuple> ZipIterator;

// This hash algorithm is taken from Thrust's monte carlo
// example.  It's needed when seeding the random number
// generator, because otherwise the results end up being
// not particularly random.
// http://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu
__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

struct ProbFunctor
{
public:
	int _seed;

	ProbFunctor(unsigned int seed) 
	{
		_seed = seed;
	}

	__host__ __device__
	tuple<int,float> operator()(ProbItem item1, ProbItem item2)
	{		
		int index1 = get<0>(item1), index2 = get<0>(item2);
		float weight1 = get<1>(item1), weight2 = get<1>(item2);

		tuple<int,float> out;
		float outWeight = weight1 + weight2;

		unsigned int seed = hash(index1 + index2 + _seed);
		default_random_engine randEngine(seed);
		uniform_real_distribution<float> uniDist(0.0f, outWeight);
	
		int outIndex = index1;
		if(uniDist(randEngine) > weight1)
			outIndex = index2;

		return tuple<int,float>(outIndex, outWeight);
	}
};

int prob_select(ZipIterator &dataBegin, ZipIterator &dataEnd, unsigned int randSeed)
{
	ProbItem initVal(-1,0.0f);
	ProbItem item = thrust::reduce(dataBegin, dataEnd, initVal, ProbFunctor(randSeed));
	return get<0>(item);
}

/*void prob_select_test(int n)
{
	cout << "Creating data" << endl;
	host_vector<ProbItem> h_data(n);
	for(int i=0; i<n; i++)
	{
		h_data[i].index = i;
		h_data[i].weight = (float)(i);
	}

	unsigned int seed = time(NULL);

	cout << "Selecting with weighted probablity" << endl;
	device_vector<ProbItem> d_data = h_data;
	int selIndex = prob_select(d_data, seed);
	cout << "Selected item is: " << selIndex << endl;
}*/

double kpp(const int n, const int m, const int k, const host_vector<float> &h_data, 
	host_vector<int> &h_outSeeds)
{
	// Copy data to device
	device_vector<float> d_data = h_data;

	// Time the function without memory copying
	clock_t start = clock();

	// Alloc mem for our seeds
	h_outSeeds.resize(k);

	// A list of indexes for our data, used for our functors
	// (0,1,2,3...n-1)
	device_vector<int> d_indices(n);
	thrust::sequence(d_indices.begin(), d_indices.end()); // O(n)

	// Alloc mem for min dists to seeds
	device_vector<float> d_minDists(n);
	// Initialize all to "infinity"
	thrust::fill(d_minDists.begin(), d_minDists.end(), FLT_MAX);

	// Pick first seed uniformly at random from data
	h_outSeeds[0] = rand()%n;
	
	// Pick the rest of the seeds probabilistically
	for(int ki=1; ki<k; ki++)
	{
		// Calculate distance of all other data points to newest seed (if smaller
		// than previous distance)
		thrust::transform(d_indices.begin(), d_indices.end(), d_minDists.begin(), 
			CalcNewMinDist(n, m, k, d_data, d_minDists, h_outSeeds[ki-1])); // O(n*m)
		
		// Zip indices and dists together
		ZipIterator begin(make_tuple(d_indices.begin(), d_minDists.begin()));
		ZipIterator end(make_tuple(d_indices.end(), d_minDists.end()));

		// Use time to seed our GPU random-functor
		unsigned int randSeed = time(NULL);

		// Use our probablistic selection to pick the next seed
		h_outSeeds[ki] = prob_select(begin, end, randSeed);
	}
	
	clock_t end = clock();
	return (double)(end-start)/CLOCKS_PER_SEC;
}

void kpp_cpu(const int n, const int m, const int k, const host_vector<float> &h_data, 
	host_vector<int> &h_outSeeds)
{
	//printf("------------------------------------------------------\n");

	// Allocate memory for arrays
	float *probs = (float*)malloc(n * sizeof(float));
	float *distsSqrd = (float*)malloc(n * sizeof(float));
	bool *nodeUsed = (bool*)malloc(n * sizeof(bool));

	//printf("Memory created for k-means++ algorithm\n");

	// Sets initial distances to the max possible
	for(int ni=0; ni<n; ni++)
		distsSqrd[ni] = FLT_MAX;

	// Set all seeds initially to -1
	for(int ki=0; ki<k; ki++)
		h_outSeeds[ki] = -1;

	//printf("Initialized arrays\n");

	// Pick the initial seed completely at random
	h_outSeeds[0] = rand() % n;

	//printf("Initial seed picked at random\n");

	// Loop until we've picked all our other seeds
	for(int ki=1; ki<k; ki++)
	{
		//printf("Finding seed for cluster #%d\n", ki+1);

		int prevSeed = h_outSeeds[ki-1];
		const float *prevSeedVec = &h_data[prevSeed*m];

		//printf("Got last seed\n");

		// Calculate the distances of each node to the previous seed
		for(int ni=0; ni<n; ni++)
		{
			float distSqrd = calcDistSq(&h_data[ni*m], prevSeedVec, m);
			if(distSqrd < distsSqrd[ni])
				distsSqrd[ni] = distSqrd;
		}

		//printf("Calculating min dists\n");

		// Calculate sum of the squared distances
		float sum = 0.0f;
		for(int ni=0; ni<n; ni++)
		{
			sum += distsSqrd[ni];
		}

		//printf("Calculated sum of min dists\n");

		// Calculate the probabilities for all the nodes based on their
		// distance squared divided by the sum of distances squared		
		for(int ni=0; ni<n; ni++)
		{
			probs[ni] = distsSqrd[ni] / sum;
			//printf("probs[%d] = %f\n", ni, probs[ni]);
		}		

		//printf("Found probabilities\n");

		// The while loop makes sure we don't use the same node twice
		while(h_outSeeds[ki] == -1 || nodeUsed[h_outSeeds[ki]] == true)
		{
			// Use the probabilities in randomly picking the next seed
			float randVal = (float)rand() / (float)RAND_MAX;
			//printf("randVal = %f\n", randVal);
			float probVal = 0.0f;
			for(int ni=0; ni<n; ni++)
			{
				float newVal = probVal + probs[ni];				
				if(randVal >= probVal && randVal < newVal)
				{
					h_outSeeds[ki] = ni;
				}
				probVal = newVal;
			}
			if(h_outSeeds[ki] == -1)
				h_outSeeds[ki] = n-1;
			
			//printf("Picked a seed to try: %d\n", outSeeds[ki]);
			//char blank[100];
			//scanf("%s", blank);
		}
		nodeUsed[h_outSeeds[ki]] = true;
		//printf("Using seed: %d\n", outSeeds[ki]);
	}	

	// Free up memory
	free(distsSqrd);
	free(probs);
	free(nodeUsed);

	//printf("Memory freed\n");
	//printf("------------------------------------------------------\n");
}

