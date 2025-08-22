#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fftw3.h"
#include "mfcc.h"

// Following https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial/notebook

void frame_audio(float* audio, int len_audio, int fft_size,
                 int frame_len, int frame_num, float** frames) {
    int len_padded = len_audio + fft_size;
    float* padded_audio = (float*)malloc((len_padded)*sizeof(float));

    // pad reflect
    int half_fft = (int) fft_size/2;
    for (int i=0; i<half_fft; ++i) {
            padded_audio[i] = audio[half_fft-i];
            padded_audio[len_audio+half_fft+i] = audio[len_audio-i-2];
    }
    for (int i=0; i<len_audio; ++i) {
        padded_audio[half_fft+i] = audio[i];
    }
    
    for (int n=0; n < frame_num; ++n) {
        for (int i=0; i < fft_size; ++i) {
            frames[n][i] = padded_audio[n*frame_len+i];
        }
    }
    free(padded_audio);
}

// Hann window
void get_hann_window(float* win, int N) {
    for (int i = 0; i < N; ++i) {
        win[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * i / (N));
    }
}

float hz_to_mel(float freq) {
    return 2595.0f * log10f(1.0f + freq / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void compute_hann_window(float** frames, int frame_num, int fft_size, float* window) {
    for (int i=0; i<frame_num; ++i) {
        for (int j=0; j<fft_size; ++j) {
            frames[i][j] = frames[i][j]*window[j];
        }
    }
}

void get_filter_points(int fmin, int fmax, int mel_filter_num, int fft_size, int sample_rate, float* mel_freqs, int* filter_points) {
    float fmin_mel = hz_to_mel(fmin);
    float fmax_mel = hz_to_mel(fmax);

    int mel_num_2 = mel_filter_num+2;
    float mel_jump = (fmax_mel - fmin_mel) / (1.0f * mel_filter_num + 1);
    for (int i=0; i<mel_num_2; ++i) {
        mel_freqs[i] = mel_to_hz(i * mel_jump);
        filter_points[i] = (int)(((fft_size + .5) / sample_rate) * mel_freqs[i]);
    }
}

void get_filters(int* filter_points, int len_filtpts, float* mel_freqs, int fft_size, float** filters) {
    // set everything to zeroes
    for (int i=0; i<len_filtpts-2; ++i) {
        for (int j=0; j<(int) (fft_size/2 + 1); ++j) {
            filters[i][j] = 0;
        }
    }
    
    for (int n=0; n<len_filtpts-2; ++n) {
        int fpn = filter_points[n];
        int fpn1 = filter_points[n+1];
        int fpn2 = filter_points[n+2];

        int diff1 = fpn1 - fpn;
        int diff1_div = diff1;
        if (diff1_div > 1) {
            diff1_div -= 1;
        }
        for (int i=0; i<diff1; ++i) {
            filters[n][fpn + i] = i*1.0f/diff1_div;
        }

        int diff2 = fpn2 - fpn1;
        int diff2_div = diff2;
        if (diff2_div > 1) {
            diff2_div -= 1;
        }
        for (int i=0; i<diff2; ++i) {
            filters[n][fpn1 + i] = 1.0f - (i*1.0f/diff2_div);
        }
    }


    // divide mel weights by width of mel band for area normalisation
    // to prevent noise increase with frequency
    for (int i=0; i<len_filtpts-2; ++i) {
        // enorm across entire row
        float enorm = 2.0 / (mel_freqs[2+i] - mel_freqs[i]);
        for (int j=0; j<(int)(fft_size/2 + 1); ++j) {
            filters[i][j] *= enorm;
        }
    }
}

void dct(int dct_filt_num, int filter_len, float** basis) {
    for (int i=0; i<filter_len; ++i) {
        basis[0][i] = 1.0 / sqrt(filter_len);
    }

    for (int i=1; i<dct_filt_num; ++i) {
        for (int j=0; j<filter_len; ++j) {
            float sample = (1 + 2 * j) * M_PI / (2.0 * filter_len);
            basis[i][j] = cosf(i * sample) * sqrtf(2.0 / filter_len);
        }
    }
}

struct mel_spectrogram* compute_melspectrogram(float* audio, int len_audio, int fft_size, int sample_rate, int hop_size,
                                                int mel_filter_num) {
    int frame_len = hop_size;
    int frame_num = (int) ((len_audio - fft_size)/(1.0f * frame_len) + 1);

    float** frames = (float**)malloc(frame_num*sizeof(float*));
    for (int i=0; i<frame_num; ++i) {
        frames[i] = (float*)malloc(fft_size*sizeof(float));
    }

    frame_audio(audio, len_audio, fft_size, frame_len, frame_num, frames);

    // run hanning window on frames
    float* hwindow = (float*)malloc(fft_size*sizeof(float));
    get_hann_window(hwindow, fft_size);
    compute_hann_window(frames, frame_num, fft_size, hwindow);
    free(hwindow);

    // now fft time!
    fftwf_complex** out = (fftwf_complex**)malloc(frame_num*sizeof(fftwf_complex*));
    for (int i=0; i<frame_num; ++i) {
        out[i] = (fftwf_complex*) fftwf_malloc(fft_size*sizeof(fftwf_complex));
        fftwf_plan plan;
        plan = fftwf_plan_dft_r2c_1d(fft_size, frames[i], out[i], FFTW_MEASURE);
        fftwf_execute(plan);

        fftwf_destroy_plan(plan);
        free(frames[i]);
    }
    free(frames);

    // compute power - for a+bi, is a^2+b^2
    // only need the low half of the fft - fft/2 + 1
    int half_fft_1 = (int) fft_size/2 + 1;
    float** power = (float**)malloc(frame_num*sizeof(float*));
    for (int i=0; i<frame_num; ++i) {
        power[i] = (float*)malloc(half_fft_1*sizeof(float));
        for (int j=0; j<half_fft_1; ++j) {
            power[i][j] = out[i][j][0]*out[i][j][0] + out[i][j][1]*out[i][j][1];
        }
        fftwf_free(out[i]);
    }
    free(out);

    // get filters for mel
    float* mel_freqs = (float*)malloc((mel_filter_num+2)*sizeof(float));
    int* filter_pts = (int*)malloc((mel_filter_num+2)*sizeof(int));

    int flow = 0;
    int fhigh = sample_rate/2;

    get_filter_points(flow, fhigh, mel_filter_num, fft_size, sample_rate, mel_freqs, filter_pts);

    // ready to construct the filters!
    int lenfiltpts = mel_filter_num+2;
    float** filters = (float**)malloc(mel_filter_num*sizeof(float*));
    for (int i=0; i<mel_filter_num; ++i) {
        filters[i] = (float*)malloc(((int)(fft_size/2 +1))*sizeof(float));
    }
    get_filters(filter_pts, lenfiltpts, mel_freqs, fft_size, filters);

    free(mel_freqs);
    free(filter_pts);

    // filter the audio - dot product - filters * audio_power(T)
    // end up with audio_filt: num_mel_filters x frame_num
    // filters is: num_mel_filters x (fft_size/2 + 1)
    // audio_power is: frame_num x (fft_size/2 + 1)
    float** audio_filtered = (float**)malloc(mel_filter_num*sizeof(float*));
    for (int i=0; i<mel_filter_num; ++i) {
        audio_filtered[i] = (float*)malloc(frame_num*sizeof(float));
        for (int j=0; j<frame_num; ++j){
            audio_filtered[i][j] = 0;
        }
    }

    // perform dot product
    for (int i=0; i<mel_filter_num; ++i) {
        for (int j=0; j<half_fft_1; ++j){
            for (int k=0; k<frame_num; ++k) {
                // power should be transposed
                audio_filtered[i][k] += filters[i][j] * power[k][j];
            }
        }
    }

    for (int i=0; i<frame_num; ++i) {
        free(power[i]);
    }
    free(power);

    for (int i=0; i<mel_filter_num; ++i) {
        free(filters[i]);
    }
    free(filters);

    struct mel_spectrogram* ms = malloc(sizeof(struct mel_spectrogram));
    ms->spectrogram = audio_filtered;
    ms->mel_filter_num = mel_filter_num;
    ms->frame_num = frame_num;

    return ms;
}

struct mfccs* compute_mfcc(struct mel_spectrogram* ms, int num_mfccs) {
    int mel_filter_num = ms->mel_filter_num;
    int frame_num = ms->frame_num;
    

    // dct
    int dct_filt_num = num_mfccs;
    float** dct_filt = (float**)malloc(dct_filt_num*sizeof(float*));
    for (int i=0; i<dct_filt_num; ++i) {
        dct_filt[i] = (float*)malloc(mel_filter_num*sizeof(float));
    }
    dct(dct_filt_num, mel_filter_num, dct_filt);

    // cepstral coeffs
    // perform dot product
    struct mfccs* result = malloc(sizeof(struct mfccs));
    result->num_mfccs = dct_filt_num;
    result->frame_num = frame_num;
    
    float** cepstral_coeffs = (float**)malloc(dct_filt_num*sizeof(float*));
    result->cepstral_coeffs = cepstral_coeffs;
    for (int i=0; i<dct_filt_num; ++i) {
        cepstral_coeffs[i] = (float*)malloc(frame_num*sizeof(float));
        for (int j=0; j<frame_num; ++j) {
            cepstral_coeffs[i][j] = 0;
        }
    }

    for (int i=0; i<dct_filt_num; ++i) {
        for (int j=0; j<frame_num; ++j) {
            for (int k=0; k<mel_filter_num; ++k) {
                cepstral_coeffs[i][j] += dct_filt[i][k] * ms->spectrogram[k][j];
            }
        }
    }

    for (int i=0; i<dct_filt_num; ++i) {
        free(dct_filt[i]);
    }
    free(dct_filt);

    return result;
}

// Computing 1st and 2nd order delta features with window size 9
#define WIN_SIZE 9
#define HALF_WIN 4

// Precomputed coefficients
float delta_coeffs[WIN_SIZE] = {
    -0.18367347, -0.12244898, -0.06122449, 0.0,
     0.06122449, 0.12244898, 0.18367347, 0.24489796, 0.30612245
};

float deltadelta_coeffs[WIN_SIZE] = {
     0.0952381, 0.015873, -0.038961, -0.077922,
    -0.101899, -0.110893, -0.104902, -0.083928, -0.047969
};

void compute_sg_derivative(float **features, float **output, int num_frames, int frame_len, int order) {
    // expecting features to come in as frame x cc, but other way so lets switch
    float* coeffs;
    if (order == 1) {
        coeffs = delta_coeffs;
    } else if (order == 2)
    {
        coeffs = deltadelta_coeffs;
    }
    
    for (int t = 0; t < num_frames; ++t) {
        for (int d = 0; d < frame_len; ++d) {
            float val = 0.0;
            for (int i = -HALF_WIN; i <= HALF_WIN; ++i) {
                int idx = t + i;
                if (idx < 0) idx = 0;
                if (idx >= num_frames) idx = num_frames - 1;
                val += coeffs[i + HALF_WIN] * features[d][idx];
            }
            output[d][t] = val;
        }
    }
}

void compute_avg_mfccs(float** mfcc, float** delta_mfcc, float** deltadelta_mfcc, float* avg_mfccs, int num_mfccs, int num_frames) {
    // avg should be (num_mfccs*3) x 1
    // mfcc is num_mfccs x num_frames
    for (int i=0; i<num_mfccs; ++i) {
        float mfcc_sum = 0;
        float mfccd_sum = 0;
        float mfccdd_sum = 0;

        for (int j=0; j<num_frames; ++j) {
            mfcc_sum += mfcc[i][j];
            mfccd_sum += delta_mfcc[i][j];
            mfccdd_sum += deltadelta_mfcc[i][j];
        }
        avg_mfccs[i] = mfcc_sum/(1.0f * num_frames);
        avg_mfccs[i+num_mfccs] = mfccd_sum/(1.0f * num_frames);
        avg_mfccs[i+2*num_mfccs] = mfccdd_sum/(1.0f * num_frames);
    }
}

void compute_melscaled(float** mel_spectrogram, int num_melfilter, int num_frames, float* mel_scaled) {
    for (int i=0; i<num_melfilter; ++i) {
        float sum = 0;
        for (int j=0; j<num_frames; ++j) {
            sum += mel_spectrogram[i][j];
        }
        mel_scaled[i] = sum/(1.0f * num_frames);
    }
}
