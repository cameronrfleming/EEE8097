// filters.h
#ifndef FILTERS
#define FILTERS

void filtfilt(double* b, double* a, double* x, int x_len, double* y);

void getButterworthLowPassCoeffs(double fcut, double fsamp, double* a, double* b);

void getButterworthHighPassCoeffs(double fcut, double fsamp, double* a, double* b);

#endif
