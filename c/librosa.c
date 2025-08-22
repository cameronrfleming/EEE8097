#include "librosa.h"
#include <sndfile.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

int min ( int a, int b ) { return a < b ? a : b; }

void load_interpolation_filter_with_ratio(char* kaiser_file, double* interp_sampled, int length, double sample_ratio) {
    FILE* fp = fopen(kaiser_file, "r");
 
    if (!fp) {
        printf("Can't open file\n");
        return;
    }
    
    // Buffer for reading in values
    char buffer[2048];

    int row = 0;
    while (row < length && fgets(buffer,2048, fp)) {
        double num = atof(buffer);

        interp_sampled[row] = num*sample_ratio;
        row++;
    }
}

void resample_loop(double* x, int x_size, double* t_out, int t_out_size, double* interp_win, int interp_win_size, 
                    double* interp_delta, int interp_delta_size, int num_table, double scale, double* y, int y_size) {

    int index_step = (int) (scale*num_table);
    double time_register = 0.0;

    int n = 0;
    double frac = 0.0;
    double index_frac = 0.0;
    int offset = 0;
    double eta = 0.0;
    double weight = 0.0;

    int nwin = interp_win_size;
    int n_orig = x_size;
    int n_out = t_out_size;

    for (int t=0; t<n_out; t++) {
        double time_register = t_out[t];

        // Top bits as index to input buff
        n = (int)time_register;

        // Grab fractional component of time index
        frac = scale * (time_register - n);

        // Offset into filter
        index_frac = frac * num_table;
        offset = (int)index_frac;

        // Interpolation factor
        eta = index_frac - offset;

        // Compute left wing of filter response
        int div = (nwin-offset) / index_step;
        int i_max = min(n+1, div);
        for (int i=0; i<i_max; i++) {
            double weight = (
                interp_win[offset + i*index_step]
                + eta*interp_delta[offset + i*index_step]
            );
            y[t] += weight * x[n-i];
        }

        // Invert P
        frac = scale - frac;

        // Offset into the filter
        index_frac = frac * num_table;
        offset = (int)index_frac;

        // Interpolation factor
        eta = index_frac - offset;

        // Compute right wing of filter response
        int k_max = min(n_orig - n - 1, (nwin - offset) / index_step);
        for (int k=0; k<k_max; k++) {
            weight = (
                interp_win[offset + k * index_step]
                + eta * interp_delta[offset + k * index_step]
            );
            y[t] += weight * x[n + k + 1];
        }
    }
}

// resampy kaiser_best implementation
void resample(double* original_data, double* resampled_data, int sr_orig, int sr_new, int n_samples_orig, int n_samples_new, bool parallel, char* kaiser_file) {
    // @todo - implement parallel
    double sample_ratio = ((double)sr_new)/sr_orig;

    // Running Ayan's code shows precision as 512
    int precision = 512;

    int data_window_length = 12289; // from the file kaiser_best_new.csv
    double interp_win[data_window_length];
    load_interpolation_filter_with_ratio(kaiser_file, interp_win, data_window_length, sample_ratio);

    // Create interpolation delta array
    double interp_delta[data_window_length];
    for (int i=0; i<data_window_length-1; i++) {
        interp_delta[i] = interp_win[i+1] - interp_win[i];
    }
    interp_delta[data_window_length-1] = 0;

    double scale = fmin(1.0, sample_ratio);
    double time_increment = 1.0/sample_ratio;

    // Create time array
    double t_out[n_samples_new];
    for (int i=0; i<n_samples_new; i++) {
        t_out[i] = i*time_increment;
    }

    resample_loop(original_data, n_samples_orig, t_out, n_samples_new, interp_win, data_window_length, interp_delta, data_window_length, precision, scale, resampled_data, n_samples_new);

}

void printfplist(double* fparray, int size, char* filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    if (!fp) {
        printf("Failed to open file %s for writing", filename);
        return;
    }

    for (int i=0; i<size; i++) {
        fprintf(fp, "%.18f\n", fparray[i]);
    }

    fclose(fp);
}

void load_soundfile(char *filename, char* kaiser_file, struct ptr_array_size_double* output_data) {
    // load the file
    // output sampling rate = 22050
    SF_INFO sfinfo;
    sfinfo.format = 0;
    SNDFILE* file = sf_open(filename, SFM_READ, &sfinfo);
    if (file == NULL) {
        printf("failed to open");
        return;
    }

    // should be wave file with signed 16bit data
    const int WAVE16 = 65538;
    if (sfinfo.format != WAVE16) {
        printf("wrong format");
        return;
    }
    if (sfinfo.channels != 1) {
        printf("Not 1 channel");
        return;
    }

    // need to load it in with reading
    double file_data[sfinfo.frames];
    sf_count_t count_ret = sf_read_double(file, file_data, sfinfo.frames);

    sf_close(file);

    // resample it
    // with 'kaiser_best'
    // using librosa and resampy implementations rewritten in c
    const double output_sr = 22050.0;
    double ratio = output_sr/sfinfo.samplerate;
    int n_samples = (int)ceil(sfinfo.frames * ratio);

    double* resampled_data = (double*)malloc(n_samples*sizeof(double));
    for (int i=0; i<n_samples; i++) {
        resampled_data[i] = 0;
    }

    resample(file_data, resampled_data, sfinfo.samplerate, output_sr, sfinfo.frames, n_samples, false, kaiser_file);

    output_data->ptr = resampled_data;
    output_data->size = n_samples;    
}

void librosa_load_and_print_soundfile(char* filename, char* outputfilename, char* kaiser_file) {
    struct ptr_array_size_double output_data;
    load_soundfile(filename, kaiser_file, &output_data);
    int n_samples = output_data.size;

    printfplist(output_data.ptr, n_samples, outputfilename);

    free(output_data.ptr);
}
