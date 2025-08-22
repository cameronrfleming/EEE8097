#ifndef PREPROCESS_DATA_H
#define PREPROCESS_DATA_H

struct feature_data {
    float* features;
    int number;
};

struct feature_data* preprocess(float* audio_data, int len_audio, int sample_rate);

struct feature_data* load_and_preprocess_soundfile(char* filename);

#endif
