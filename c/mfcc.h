#ifndef MFCC_H
#define MFCC_H

struct mfccs {
    float** cepstral_coeffs;
    int num_mfccs;
    int frame_num;
};

struct mel_spectrogram{
    float** spectrogram;
    int mel_filter_num;
    int frame_num;
};

void frame_audio(float* audio, int len_audio, int fft_size,
    int frame_len, int frame_num, float** frames);

struct mel_spectrogram* compute_melspectrogram(float* audio, int len_audio, 
    int fft_size, int sample_rate, int hop_size, int mel_filter_num);

struct mfccs* compute_mfcc(struct mel_spectrogram* ms,
    int num_mfccs);

void compute_sg_derivative(float **features, float **output, int num_frames,
    int frame_len, int order);

void compute_avg_mfccs(float** mfcc, float** delta_mfcc, float** deltadelta_mfcc,
    float* avg_mfccs, int num_mfccs, int num_frames);

void compute_melscaled(float** mel_spectrogram, int num_melfilter,
    int num_frames, float* mel_scaled);

#endif
