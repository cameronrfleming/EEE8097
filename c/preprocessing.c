#include "librosa.h"
#include "preprocess_data.h"
#include "extract_audiofiles.h"
#include "filters.h"
#include "binarize.h"
#include "train_and_output_model.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

const int NUM_LABELS = 10;

struct split_data {
    char** training_filenames;
    int* training_categories;
    char** test_filenames;
    int* test_categories;
    int num_training;
    int num_test;
};

long currentMillis() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
	// Convert the seconds to milliseconds by multiplying by 1000
	// Convert the microseconds to milliseconds by dividing by 1000
}

void shuffle(int *array, int n, int num_shuffles) {
    srand((unsigned)time(NULL));
    for (int j = 0; j < num_shuffles; j++) {
        for (int i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

struct split_data* split_train_test(char** audio_files, int* enc_categories, int percent_test) {
    // get the numbers and shuffle them
    int* indices = (int*)malloc(NUM_AUDIO_FILES*sizeof(int));
    for (int i=0; i<NUM_AUDIO_FILES; ++i) {
        indices[i] = i;
    }
    shuffle(indices, NUM_AUDIO_FILES, 10);

    int num_test = (int)NUM_AUDIO_FILES*(percent_test/100.0f);
    int num_training = NUM_AUDIO_FILES - num_test;
    printf("%d Test\n%d Training\n", num_test, num_training);

    // Just point to where they already are
    char** training_files = (char**)malloc(num_training*sizeof(char*));
    int* training_categories = (int*)malloc(num_training*sizeof(int)); 
    char** test_files = (char**)malloc(num_test*sizeof(char*));
    int* test_categories = (int*)malloc(num_test*sizeof(int));

    for (int i=0; i<num_training; ++i) {
        training_files[i] = audio_files[indices[i]];
        training_categories[i] = enc_categories[indices[i]];
    }
    for (int i=0; i<num_test; ++i) {
        test_files[i] = audio_files[indices[i+num_training]];
        test_categories[i] = enc_categories[indices[i+num_training]];
    }
    free(indices);

    struct split_data* sd = (struct split_data*)malloc(sizeof(struct split_data));
    sd->training_filenames = training_files;
    sd->training_categories = training_categories;
    sd->test_filenames = test_files;
    sd->test_categories = test_categories;
    sd->num_training = num_training;
    sd->num_test = num_test;
    return sd;
}

void encode_labels(char** categories, char** labels, int* encoded_labels) {
    int label_loc = 0;
    for (int i=0; i<NUM_AUDIO_FILES; ++i) {
        if (label_loc < NUM_LABELS) {
            int found = 0;
            for (int j=0; j<label_loc; ++j) {
                if (strcmp(labels[j], categories[i]) == 0) {
                    encoded_labels[i] = j;
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                sprintf(labels[label_loc], "%s", categories[i]);
                encoded_labels[i] = label_loc;
                label_loc += 1;
            }
        } else {
            int found = 0;
            for (int j=0; j<NUM_LABELS; ++j) {
                if (strcmp(labels[j], categories[i]) == 0) {
                    encoded_labels[i] = j;
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                printf("ERROR: %d = %s\n", i, categories[i]);
            }
        }
    }
}

int main() {
    time_t time_start = time(NULL);
    long t_ms_s = currentMillis();

    // load all the files
    char meta_data_path[] = "../tsetlin/env_data/ESC-50-master/meta/esc50.csv";
    struct filedata* filed = filter_audio_files(meta_data_path);

    time_t time_filteraudio = time(NULL);
    printf("Time elapsed filter audio: %ld\n", time_filteraudio - time_start);

    // Get labels
    char** labels = (char**)malloc(NUM_LABELS*sizeof(char*));
    for (int i=0; i<NUM_LABELS; ++i) {
        labels[i] = (char*)malloc(50*sizeof(char));
    }
    int* encoded_labels = (int*)malloc(NUM_AUDIO_FILES*sizeof(int));
    encode_labels(filed->categories, labels, encoded_labels);

    time_t time_labels = time(NULL);
    printf("Time elapsed labels: %ld\n", time_labels - time_start);

    // split into training vs test
    struct split_data* sd = split_train_test(filed->audio_files, encoded_labels, 20);

    // free filed
    for (int i=0; i<NUM_AUDIO_FILES; ++i) {
        free(filed->categories[i]);
    }
    // audio files are pointed to
    free(filed->categories);
    free(filed->audio_files);
    free(filed);
    free(encoded_labels);

    time_t time_splitdata = time(NULL);
    printf("Time elapsed split data: %ld\n", time_splitdata - time_start);
    
    // loop through and preprocess the training data to get features
    float** list_training_feature_data = (float**)malloc(sd->num_training*sizeof(float*));
    int num_features = 0;
    for (int i=0; i<sd->num_training; ++i) {
        struct feature_data* featd = load_and_preprocess_soundfile(sd->training_filenames[i]);
        list_training_feature_data[i] = featd->features;
        free(sd->training_filenames[i]);
        if (i == 0) {
            num_features = featd->number;
        } else if (num_features != featd->number) {
            printf("ERROR IN NUM FEATURES %d\n", i);
        }
        free(featd);
    }

    time_t time_preprocess = time(NULL);
    printf("Time elapsed preprocess: %ld\n", time_preprocess - time_start);
    
    // Standardise TRAINING data only
    float* means = (float*)malloc(num_features*sizeof(float));
    float* stddevs = (float*)malloc(num_features*sizeof(float));
    compute_mean_stddevs_and_standardise(list_training_feature_data, sd->num_training, num_features, means, stddevs);

    time_t time_standardise = time(NULL);
    printf("Time elapsed standardise: %ld\n", time_standardise - time_start);
    
    // binarize
    int resolution = 25;
    // ONLY NEED BITS - using int8 for now
    int8_t** x_train_binarized = (int8_t**)malloc(sd->num_training*sizeof(int8_t*));
    for (int i=0; i<sd->num_training; ++i) {
        x_train_binarized[i] = (int8_t*)malloc(num_features*resolution*sizeof(int8_t));
    }
    float** thresholds = (float**)malloc(num_features*sizeof(float*));
    for (int i=0; i<num_features; ++i) {
        thresholds[i] = (float*)malloc(resolution*sizeof(float));
    }
    get_thresholds_and_binarize(list_training_feature_data, sd->num_training, num_features, resolution, x_train_binarized, thresholds);

    time_t time_binarize = time(NULL);
    printf("Time elapsed Binarize: %ld\n", time_binarize - time_start);

    for (int i=0; i<sd->num_training; ++i){
        free(list_training_feature_data[i]);
    }
    free(list_training_feature_data);

    for (int i=0; i<NUM_LABELS; ++i) {
        free(labels[i]);
    }
    free(labels);

    // training features: print to a file
    char* filenameO = "data/txt_c_train_features.txt";
    FILE * fp;
    fp = fopen(filenameO, "w");

    fprintf(fp, "%d\n", sd->num_training);
    fprintf(fp, "%d\n", num_features*resolution);
    fprintf(fp, "%d\n", NUM_LABELS);

    // Inputs needed for MCTM
    for (int i=0; i<sd->num_training; ++i) {
        for (int j=0; j<num_features*resolution; ++j) {

            fprintf(fp, "%d ", x_train_binarized[i][j]);
        }
        fprintf(fp, "%d\n", sd->training_categories[i]);
    }
    fclose(fp);

    long t_ms_pp = currentMillis();
    printf("Time elapsed preprocess total: %f\n", (t_ms_pp - t_ms_s)/1000.0);

    // preprocess test data
    float** list_test_feature_data = (float**)malloc(sd->num_test*sizeof(float*));
    for (int i=0; i<sd->num_test; ++i) {
        struct feature_data* featd = load_and_preprocess_soundfile(sd->test_filenames[i]);
        list_test_feature_data[i] = featd->features;
        free(sd->test_filenames[i]);
        if (i == 0) {
            num_features = featd->number;
        } else if (num_features != featd->number) {
            printf("ERROR IN NUM FEATURES %d\n", i);
        }
        free(featd);
    }

    time_t time_preprocesst = time(NULL);
    printf("Time elapsed preprocess test: %ld\n", time_preprocesst - time_start);
    // Standardise TEST data only
    standardise_data(list_test_feature_data, sd->num_test, num_features, means, stddevs);

    time_t time_standardiset = time(NULL);
    printf("Time elapsed standardise test: %ld\n", time_standardiset - time_start);
    
    // binarize
    int8_t** x_test_binarized = (int8_t**)malloc(sd->num_test*sizeof(int8_t*));
    for (int i=0; i<sd->num_test; ++i) {
        x_test_binarized[i] = (int8_t*)malloc(num_features*resolution*sizeof(int8_t));
    }

    binarize_features(list_test_feature_data, x_test_binarized, thresholds, sd->num_test, num_features, resolution);

    for (int i=0; i<sd->num_test; ++i){
        free(list_test_feature_data[i]);
    }
    free(list_test_feature_data);

    time_t time_binarizet = time(NULL);
    printf("Time elapsed Binarize test: %ld\n", time_binarizet - time_start);

    long t_ms_test = currentMillis();
    printf("Time elapsed test total: %f\n", (t_ms_test - t_ms_pp)/1000.0);

    // test features: print to a file
    char* filename1 = "data/txt_c_test_features.txt";
    FILE * fp1;
    fp1 = fopen(filename1, "w");

    fprintf(fp1, "%d\n", sd->num_test);
    fprintf(fp1, "%d\n", num_features*resolution);
    fprintf(fp1, "%d\n", NUM_LABELS);

    // Inputs needed for MCTM
    for (int i=0; i<sd->num_test; ++i) {
        for (int j=0; j<num_features*resolution; ++j) {

            fprintf(fp1, "%d ", x_test_binarized[i][j]);
        }
        fprintf(fp1, "%d\n", sd->test_categories[i]);
    }
    fclose(fp1);

    for (int i=0; i<num_features; ++i) {
        free(thresholds[i]);
    }
    free(thresholds);
    free(means);
    free(stddevs);

    for (int i=0; i<sd->num_training; ++i) {
        free(x_train_binarized[i]);
    }
    free(x_train_binarized);
    for (int i=0; i<sd->num_test; ++i) {
        free(x_test_binarized[i]);
    }
    free(x_test_binarized);
    for (int i=0; i<sd->num_test; ++i) {
        free(sd->test_filenames[i]);
    }
    free(sd->training_filenames);
    free(sd->test_filenames);
    free(sd->training_categories);
    free(sd->test_categories);
    free(sd);
    return 0;
