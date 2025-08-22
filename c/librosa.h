// librosa.h
#ifndef LIBROSA
#define LIBROSA

struct ptr_array_size_double {
    double* ptr;
    int size;
};

void load_soundfile(char* filename, char* kaiser_file, struct ptr_array_size_double* output_data);

#endif
