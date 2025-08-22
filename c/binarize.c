#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "binarize.h"

/*
    Gets thresholds for binarization
    m - number of test samples
    n - number of features
    Inputs: 
        resolution - number of possibilities
        x_train_scaled - training data that has been scaled - m x n
        thresholds - output of the thresholds - n x resolution
        m - number of test samples
        n - number of features
*/
void get_binarization_thresholds(int resolution, float** x_train_scaled, float** thresholds, int m, int n) {
    float* feature_mins = (float*)malloc(sizeof(float)*n);
    float* feature_maxs = (float*)malloc(sizeof(float)*n);

    // Initialise the mins and maxs
    for (int j=0; j<n; ++j) {
        feature_mins[j] = x_train_scaled[0][j];
        feature_maxs[j] = x_train_scaled[0][j];
    }

    // Loop through to find actual min and max
    for (int i=1; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            if (x_train_scaled[i][j] < feature_mins[j]) {
                feature_mins[j] = x_train_scaled[i][j];
            }

            if (x_train_scaled[i][j] > feature_maxs[j]) {
                feature_maxs[j] = x_train_scaled[i][j];
            }
        }
    }

    // Loop through and set the thresholds
    for (int j=0; j<n; ++j) {
        for (int k=0; k<resolution; ++k) {
            float step_size = (feature_maxs[j] - feature_mins[j])/(1.0f * resolution);
            thresholds[j][k] = feature_mins[j] + k*step_size;
        }
    }

    free(feature_mins);
    free(feature_maxs);
}

/* 
    Binarizes features.
    m - number of samples
    n - number of features
    res - resolution of binarization

    Inputs:
        X - sample input, m x n
        X_binarized - binarized output of X, m x (n*res)
        thresholds - thresholds of the values for binarization, n x res
        m - number of samples
        n - number of features
        res - resolution

*/
void binarize_features(float** X, int8_t** X_binarized, float** thresholds, int m, int n, int res) {
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            for (int k=0; k<res; ++k) {
                if (X[i][j] > thresholds[j][k]) {
                    X_binarized[i][j*res + k] = 1;
                } else {
                    X_binarized[i][j*res + k] = 0;
                }
            }
        }
    }
}

/*
    x_train should be m x n where m is num samples, n is num features
*/
void get_thresholds_and_binarize(float** x_train, int num_samples, int num_features, int resolution, int8_t** x_binarized, float** thresholds) {
    get_binarization_thresholds(
        resolution,
        x_train,
        thresholds,
        num_samples,
        num_features
    );

    binarize_features(
        x_train,
        x_binarized,
        thresholds,
        num_samples,
        num_features,
        resolution
    );
}

/* 
    Computes the means and std deviations and standardises features
*/
void compute_mean_stddevs_and_standardise(float** features, int num_samples, int num_features, float* means, float* stddevs) {
    // compute means first
    for (int i=0; i<num_features; ++i) {
        float sum = 0.0;
        for (int j=0; j<num_samples; ++j) {
            sum += features[j][i];
        }
        means[i] = sum/(1.0*num_samples);
    }

    // compute std deviations using means
    for (int i=0; i<num_features; ++i) {
        float values = 0.0;
        for (int j=0; j<num_samples; ++j) {
            values += pow(features[j][i] - means[i], 2);
        }

        values = values/num_samples;
        stddevs[i] = sqrt(values);
    }

    standardise_data(features, num_samples, num_features, means, stddevs);
}

/*
    Given computed means and stddevs, standardise the features
*/
void standardise_data(float** features, int num_samples, int num_features, float* means, float* stddevs) {
    // standardise it
    for (int i=0; i<num_features; ++i) {
        for (int j=0; j<num_samples; ++j) {
            features[j][i] = (features[j][i] - means[i])/stddevs[i];
        }
    }
}

