#ifndef BINARIZE_H
#define BINARIZE_H

#include <stdint.h>

void get_thresholds_and_binarize(float** x_train, int num_samples, int num_features, int resolution, int8_t** x_binarized, float** thresholds);

void standardise_data(float** features, int num_samples, int num_features, float* means, float* stddevs);

void compute_mean_stddevs_and_standardise(float** features, int num_samples, int num_features, float* means, float* stddevs);

void binarize_features(float** X, int8_t** X_binarized, float** thresholds, int m, int n, int res);

#endif

