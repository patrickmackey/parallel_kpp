#pragma once

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace thrust;

/** Tests our probablistic selection algorithm */
void prob_select_test(int n);

/**
 * Perform k-means++ initialization on an Nvidia GPU using the Thrust library.
 * @param n  Number of data points.
 * @param m  Dimension of data.
 * @param k  Number of clusters.
 * @param h_data  Data to be clustered (assumes to be in flattened matrix form).
 * @param h_seeds  Output: The data points to use as the initial seeds.
 */
double kpp(const int n, const int m, const int k, const host_vector<float> &h_data,
		 host_vector<int> &h_outSeeds);

/**
 * Perform k-means++ initialization on a CPU using a single thread.
 * @param n  Number of data points.
 * @param m  Dimension of data.
 * @param k  Number of clusters.
 * @param h_data  Data to be clustered (assumes to be in flattened matrix form).
 * @param h_seeds  Output: The data points to use as the initial seeds.
 */
void kpp_cpu(const int n, const int m, const int k, const host_vector<float> &h_data,
		 host_vector<int> &h_outSeeds);
