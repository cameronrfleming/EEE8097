#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "preprocess_data.h"
#include "mfcc.h"
#include "librosa.h"
#include "filters.h"

struct feature_data* preprocess(float* audio_data, int len_audio, int sample_rate) {
    int fft_size = 2048;
    int hop_size = 512;

    // run scaled melspectrogram
    int num_mels_mel = 64;
    float* mel_scaled = (float*)malloc(num_mels_mel*sizeof(float));
    struct mel_spectrogram* ms1 = compute_melspectrogram(audio_data, len_audio, fft_size, sample_rate, hop_size, num_mels_mel);
    compute_melscaled(ms1->spectrogram, ms1->mel_filter_num, ms1->frame_num, mel_scaled);

    // run mfccs
    int num_mels_mfccs = 128;
    int num_mfccs = 13;
    struct mel_spectrogram* ms2 = compute_melspectrogram(audio_data, len_audio, fft_size, sample_rate, hop_size, num_mels_mfccs);
    struct mfccs* mfcc = compute_mfcc(ms2, num_mfccs);
    float** d_mfcc = (float**)malloc(num_mfccs*sizeof(float*));
    float** dd_mfcc = (float**)malloc(num_mfccs*sizeof(float*));
    for (int i=0; i<num_mfccs; ++i) {
        d_mfcc[i] = (float*)malloc(mfcc->frame_num * sizeof(float));
        dd_mfcc[i] = (float*)malloc(mfcc->frame_num * sizeof(float));
    }
    compute_sg_derivative(mfcc->cepstral_coeffs, d_mfcc, mfcc->frame_num, mfcc->num_mfccs, 1);
    compute_sg_derivative(mfcc->cepstral_coeffs, dd_mfcc, mfcc->frame_num, mfcc->num_mfccs, 2);
    float* avg_mfccs = (float*)malloc(3*num_mfccs*sizeof(float));
    compute_avg_mfccs(mfcc->cepstral_coeffs, d_mfcc, dd_mfcc, avg_mfccs, num_mfccs, mfcc->frame_num);

    // concatenate features
    struct feature_data* fd = (struct feature_data*)malloc(sizeof(struct feature_data));
    fd->number = num_mels_mel+(num_mfccs*3);
    fd->features = (float*)malloc((fd->number)*sizeof(float));
    for (int i=0; i<num_mels_mel; ++i) {
        fd->features[i] = mel_scaled[i];
    }
    for (int i=0; i<num_mfccs*3; ++i) {
        fd->features[i+num_mels_mel] = avg_mfccs[i];
    }

    // free other data ms and mfcc stuff
    free(mel_scaled);
    for (int i=0; i<ms1->mel_filter_num; ++i) {
        free(ms1->spectrogram[i]);
    }
    free(ms1->spectrogram);
    free(ms1);
    for (int i=0; i<ms2->mel_filter_num; ++i) {
        free(ms2->spectrogram[i]);
    }
    free(ms2->spectrogram);
    free(ms2);
    for (int i=0; i<num_mfccs; ++i) {
        free(mfcc->cepstral_coeffs[i]);
        free(d_mfcc[i]);
        free(dd_mfcc[i]);
    } 
    free(mfcc->cepstral_coeffs);
    free(mfcc);
    free(d_mfcc);
    free(dd_mfcc);
    free(avg_mfccs);

    return fd;
}

struct feature_data* load_and_preprocess_soundfile(char* filename) {
    char kaiser_file[32] = "kaiser_best_new.csv";
    char full_filename[150] = "../tsetlin/env_data/ESC-50-master/audio/";
    strcat(full_filename, filename);
    struct ptr_array_size_double output_data;
    load_soundfile(full_filename, kaiser_file, &output_data);

    // Filtering - get coefficients
    double fsamp = 22050;
    double a_l[3];
    double b_l[3];
    int lp_fcutoff = 4000;
    getButterworthLowPassCoeffs(lp_fcutoff, fsamp, a_l, b_l);
    double a_h[3];
    double b_h[3];
    int hp_fcutoff = 300;
    getButterworthHighPassCoeffs(hp_fcutoff, fsamp, a_h, b_h);

    // Filtering - running low and high pass filters
    double* lp_output = (double*)malloc(output_data.size*sizeof(double));
    for (int i=0; i<output_data.size; i++) {
        lp_output[i] = 0;
    }
    filtfilt(b_l, a_l, output_data.ptr, output_data.size, lp_output);

    // Reuse output data for high pass output
    for (int i=0; i<output_data.size; i++) {
        output_data.ptr[i] = 0;
    }

    // High pass
    filtfilt(b_h, a_h, lp_output, output_data.size, output_data.ptr);
    free(lp_output);

    float* data_flt = (float*)malloc(output_data.size*sizeof(float));
    for (int i = 0; i < output_data.size; ++i) {
        data_flt[i] = (float)output_data.ptr[i];
    }
    free(output_data.ptr);

    struct feature_data* fd = preprocess(data_flt, output_data.size, fsamp);
    free(data_flt);
    return fd;
}
