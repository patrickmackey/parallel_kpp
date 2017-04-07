#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace thrust;
using namespace std;

/**
 * Perform KKZ initialization on an Nvidia GPU using the Thrust library.
 * @param n  Number of data points.
 * @param m  Dimension of data.
 * @param k  Number of clusters.
 * @param h_data  Data to be clustered (assumes to be in flattened matrix form).
 * @param h_seeds  Output: The data points to use as the initial seeds.
 */
double kkz(const int n, const int m, const int k, const host_vector<float> &h_data,
		 host_vector<int> &h_outSeeds);

/**
* Perform KKZ initialization on an Nvidia GPU using CUDA instead of thrust for everything except reductions
* @param n  Number of data points.
* @param m  Dimension of data.
* @param k  Number of clusters.
* @param h_data  Data to be clustered (assumes to be in flattened matrix form).
* @param h_seeds  Output: The data points to use as the initial seeds.
*/
double kkz_cuda(const int n, const int m, const int k, const host_vector<float> &h_data,
		host_vector<int> &h_outSeeds);

/**
 * Perform KKZ initialization on a CPU using a single thread.
 * @param n  Number of data points.
 * @param m  Dimension of data.
 * @param k  Number of clusters.
 * @param h_data  Data to be clustered (assumes to be in flattened matrix form).
 * @param h_seeds  Output: The data points to use as the initial seeds.
 */
void kkz_cpu(const int n, const int m, const int k, const host_vector<float> &h_data,
		 host_vector<int> &h_outSeeds);

/**
 * Perform KKZ initialization on a CPU using a single thread.  (Faster implementation).
 * @param n  Number of data points.
 * @param m  Dimension of data.
 * @param k  Number of clusters.
 * @param h_data  Data to be clustered (assumes to be in flattened matrix form).
 * @param h_seeds  Output: The data points to use as the initial seeds.
 */
void kkz_cpu2(const int n, const int m, const int k, const host_vector<float> &h_data,
		 host_vector<int> &h_outSeeds);
