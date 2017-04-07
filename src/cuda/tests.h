#ifndef TESTS_H
#define TESTS_H

/**
 * Performs tests for KKZ initialization using random values.
 * @param n  Number of data points.
 * @param m  Number of dimensions.
 * @param k  Number of clusters.
 * @param numTrials  Number of trials.
 */
void kkz_test_new(int n, int m, int k, int numTrials);

/**
 * Performs tests for k-means++ initialization using random values.
 * @param n  Number of data points.
 * @param m  Number of dimensions.
 * @param k  Number of clusters.
 * @param numTrials  Number of trials.
 */
void kpp_test_new(int n, int m, int k, int numTrials);

/**
 * Performs a test for KKZ initialization using random values.
 * @param n  Number of data points.
 * @param m  Number of dimensions.
 * @param k  Number of clusters.
 */
void kkz_test(int n, int m, int k);

/**
 * Performs a test for k-means++ initialization using random values.
 * @param n  Number of data points.
 * @param m  Number of dimensions.
 * @param k  Number of clusters.
 */
void kpp_test(int n, int m, int k);

#endif
