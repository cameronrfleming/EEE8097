#include "filters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double factor = 1.0;

typedef struct 
{
    double real;
    double imag;
} imaginary_num;

void getButterworthLowPassCoeffs(double fcut, double fsamp, double* a, double* b )
{
    double pi = 4.0*atan(1.0);
    double sqrt2 = sqrt(2.0);

    double alpha = 1.0 / tan( factor * pi * fcut / fsamp );
    b[0] = 1.0 / ( 1.0 + sqrt2 * alpha + alpha * alpha );
    b[1] = 2 * b[0];
    b[2] = b[0];
    a[0] = 1.0;
    a[1] = 2.0 * ( 1.0 - alpha * alpha ) * b[0];
    a[2] = ( 1.0 - sqrt2 * alpha + alpha * alpha ) * b[0];
}

double compute_real(double real, double imag) {
    double num = (4+real)*(4-real) - imag*imag;
    double denom = (4-real)*(4-real) + imag*imag;

    return num/denom;
}

double compute_imag(double real, double imag) {
    double num = (4+real)*imag + (4-real)*imag;
    double denom = (4-real)*(4-real) + imag*imag;
    
    return num/denom;
}


void getButterworthHighPassCoeffs(double fcut, double fsamp, double* a, double* b) {
    double polevals = sqrt(2)/2;
    imaginary_num p[2];
    p[0].real = -polevals;
    p[0].imag = polevals;
    p[1].real = -polevals;
    p[1].imag = -polevals; 

    double k = 1;
    double Wn = fcut/(fsamp/2);
    double fs = 2.0;
    double warped = 2 * fs * tan(M_PI*Wn/fs);
    double wo = warped;
    const int degree = 2;

    imaginary_num p_hp[2];
    p_hp[0].real = wo/(2*p[0].real);
    p_hp[0].imag = -wo/(2*p[0].imag);
    p_hp[1].real = wo/(2*p[1].real);
    p_hp[1].imag = -wo/(2*p[1].imag); 

    double z_hp[degree] = {0, 0};
    double k_hp = 1;

    // bilinear_zpk
    int degree_hp = 0;
    double z_z[2] = {(fs*2 + z_hp[0])/(fs*2 - z_hp[0]), (fs*2 + z_hp[1])/(fs*2 - z_hp[1])};

    imaginary_num p_z[2];
    p_z[0].real = compute_real(p_hp[0].real, p_hp[0].imag);
    p_z[0].imag = compute_imag(p_hp[0].real, p_hp[0].imag);
    p_z[1].real = compute_real(p_hp[1].real, p_hp[1].imag);
    p_z[1].imag = compute_imag(p_hp[1].real, p_hp[1].imag);
    
    double k_z = k_hp*16/((4-p_hp[0].real)*(4-p_hp[1].real)-p_hp[0].imag*p_hp[1].imag);

    b[0] = 1 * k_z;
    b[1] = -2 * k_z;
    b[2] = 1 * k_z;

    a[0] = 1;
    a[1] = -(p_z[0].real + p_z[1].real);
    a[2] = p_z[0].real*p_z[1].real - p_z[0].imag*p_z[1].imag;
}

// initial conditions
// assume 2nd order filter so b and a are of size 3
void lfilter_zi(double* b, double* a, double* zi) {
    // if a[0] != 1 error
    if (a[0] != 1.0) {
        fprintf(stderr, "a[0] should be 1");
    }

    double eye[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    double companion[2][2] = {{-a[1]/a[0], 1}, {-a[2]/a[0], 0}};
    double IminusA[2][2];
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            IminusA[i][j] = eye[i][j] - companion[i][j];
        }
    }

    double B[2] = {b[1] - a[1]*b[0], b[2] - a[2]*b[0]};

    zi[0] = (B[0] + B[1])/(IminusA[0][0] + IminusA[1][0]);
    zi[1] = (1.0+a[1])*zi[0] - (b[1] - a[1]*b[0]);
}

void padfiltfilt(int padding, double* x, int x_len, double** xnew) {
    double left_end = x[0];
    double* left_slice = (double*) malloc(padding * sizeof(double));
    for(int i=padding; i>0; i--) {
        left_slice[padding-i] = x[i];
    }

    double right_end = x[x_len-1];
    double* right_slice = (double*) malloc(padding * sizeof(double));
    for(int i=x_len-2; i>x_len-(padding+2); i--) {
        right_slice[x_len-(i+2)] = x[i];
    }

    *xnew = (double*) malloc((x_len+padding+padding) * sizeof(double));

    for(int i=0; i<padding; i++) {
        (*xnew)[i] = 2*left_end - left_slice[i];
    }
    for(int i=padding; i<x_len+padding; i++) {
        (*xnew)[i] = x[i-padding];
    }
    for(int i=x_len+padding; i<x_len+padding+padding; i++) {
        (*xnew)[i] = 2*right_end - right_slice[i-(x_len+padding)];
    }

    // FREE THE MEM!!!
    free(left_slice);
    free(right_slice);
}

void filter(double* b, double* a, double* x, int x_len, double* y, double* zi) {
    int filter_order = 3;

    for(int i=0; i<x_len; ++i) {
        int order = filter_order-1;
        while (order) {
            if(i >= order) {
                zi[order-1] = b[order]*x[i-order] - a[order]*y[i-order] +zi[order];
            }
            --order;
        }
        y[i] = b[0]*x[i] + zi[0];
    }
}

void reverse_array(double* arr, int length) {
    for (int i=0; i<length/2; i++) {
        double temp = arr[i];
        arr[i] = arr[length-i-1];
        arr[length-i-1] = temp;
    }
}

void filtfilt(double* b, double* a, double* x, int x_len, double* y) {
    // steady-state of filter step resp
    double zi[2];
    lfilter_zi(b, a, zi);

    // pad filter
    int padding = 9;
    double* xnew;
    padfiltfilt(padding, x, x_len, &xnew);
    int xnew_len = x_len+padding+padding;

    // forward filter
    double x0 = xnew[0];
    double* y1 = (double*) malloc(xnew_len * sizeof(double));
    double zi1[3] = {zi[0]*x0, zi[1]*x0, 0};
    filter(b, a, xnew, xnew_len, y1, zi1);

    // free xnew now
    free(xnew);

    // reverse filter
    reverse_array(y1, xnew_len);
    double y0 = y1[0];
    double* y2 = (double*) malloc(xnew_len * sizeof(double));
    double zi2[3] = {zi[0]*y0, zi[1]*y0, 0};
    filter(b, a, y1, xnew_len, y2, zi2);
    reverse_array(y2, xnew_len);

    // remove padding
    for(int i=0; i<x_len; i++) {
        y[i] = y2[padding+i];
    }

    // clean up xnew, y1
    free(y1);
    free(y2);
}

