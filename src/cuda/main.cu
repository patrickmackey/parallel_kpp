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

void dispHelp(const char *exe)
{
	printf("To use %s, used must enter the following variables:\n", exe);
	printf("%s [algorithm] [n] [m] [k]\n", exe);
	printf("Where:\n");
	printf("      algorithm can be:\n");
	printf("           1 for kkz\n");
	printf("           2 for k-means++\n");
	printf("      n = number of data points\n");
	printf("      m = dimension of each data point\n");
	printf("      k = number of clusters desired\n");
}

int main(int argc, char **argv)
{
	// Make sure it has the correct number of arguments
	if(argc < 5)
	{
		dispHelp(argv[0]);
		return -1;
	}

	// Get variables from command line
	int alg = atoi(argv[1]);
	int n = atoi(argv[2]);
	int m = atoi(argv[3]);
	int k = atoi(argv[4]);
	int numTrials = 10;
	
	// Make sure arguments are acceptable
	if(n < 1 || m < 1 || k < 2 || alg < 1 || alg > 2)
	{
		dispHelp(argv[0]);
		return -1;
	}

	//printf("\nUsing %d trials\n\n", numTrials);

	// Perform test based on algorithm selected
	if(alg == 1)
	{		
		kkz_test_new(n, m, k, numTrials);		
	}
	else
	{
		kpp_test_new(n, m, k, numTrials);
	}
	return 0;
}
