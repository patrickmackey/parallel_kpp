#ifndef KMEANS__H
#define KMEANS__H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace thrust;

/**
 * Performs k-means clustering on an Nvidia GPU using the Thrust library.
 * @param n  Number of data points.
 * @param m  Dimension of data.
 * @param k  Number of clusters.
 * @param h_data  Data to be clustered (assumes to be in flattened matrix form).
 * @param h_seeds  Initial seeds for the clustering process.
 * @param h_membership  The output results from the clustering process.  Stores the cluster each data point was assigned to.
 */
void kmeans_thrust(const int n, const int m, const int k,
	const host_vector<float> &h_data, const host_vector<int> &h_seeds,
	host_vector<int> &h_membership);

#endif
