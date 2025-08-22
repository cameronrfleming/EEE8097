#ifndef EXTRACT_AUDIOFILES_H
#define EXTRACT_AUDIOFILES_H

extern const int NUM_AUDIO_FILES;

struct filedata {
    char** audio_files;
    char** categories;
};

struct filedata* filter_audio_files(char *meta_data_path);

#endif
