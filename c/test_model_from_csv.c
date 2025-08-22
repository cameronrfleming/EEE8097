#include "IndexedTsetlinMachine.h"
#include "build_model_from_csv.h"
#include "Tools.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>



long currentMillis() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

int main(int argc, char* argv[]) {
    if (argc > 2) {
        return 1;
    }

    char* input_filename;
    if (argc == 1) {
        input_filename = "data/txt_cpp_features_test.txt";
        printf("Running default CPP version: %s\n", input_filename);
    } else if (strcmp(argv[1], "delta") == 0) {
        input_filename = "data/txt_cpp_features_test_delt.txt";
        printf("Running delta CPP version: %s\n", input_filename);
    } else if (strcmp(argv[1], "py") == 0) {
        input_filename = "data/txt_py_test_output_feat.txt";
        printf("Running Python version: %s\n", input_filename); 
    } else if (strcmp(argv[1], "c") == 0) {
        input_filename = "data/txt_c_test_features.txt";
        printf("Running C version: %s\n", input_filename);
    } else if (strcmp(argv[1], "xor") == 0) {
        input_filename = "data/NoisyXORTestData.txt";
        printf("Running XOR version: %s\n", input_filename);
    } else {
        return 1;
    }
    
    time_t time_start = time(NULL);
    long t_b_s = currentMillis();

    // Build the model
    char* model_filename = "data/txt_model_output.txt";

    int append_negated;
    struct IndexedTsetlinMachine* itm = build_indexedTM_model_from_CSV(model_filename, &append_negated);

    time_t time_buildmodel = time(NULL);
    long t_b_e = currentMillis();
    printf("Time elapsed rebuild model: %ld\n", time_buildmodel - time_start);
    printf("Time elapsed rebuild model: %f\n", (t_b_e - t_b_s)/1000.0);

    FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(input_filename, "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

    getline(&line, &len, fp);
    int num_test_data = atoi(line);
    getline(&line, &len, fp);
    int num_features = atoi(line);
    getline(&line, &len, fp);
    int num_classes = atoi(line);

    int num_samples = num_test_data;
    unsigned int* X_test = (unsigned int*)malloc(num_samples*num_features*sizeof(unsigned int));
    int* y_test = (int*)malloc(num_samples*sizeof(int));

    read_samples_file(fp, num_samples, num_features, X_test, y_test);

    int* y_pred = (int*)malloc(num_test_data*sizeof(int));
    int num_ta_chunks = (int) (2*num_features - 1)/32 + 1;
    int len_encoded_x_test = num_test_data*num_ta_chunks;
    
    unsigned int* encoded_X_test = (unsigned int*)malloc(len_encoded_x_test*sizeof(unsigned int));
    tm_encode(X_test, encoded_X_test, num_test_data, num_features, 1, 1, num_features, 1, append_negated, 0);

    // run one test and time it
    unsigned int* enc1 = (unsigned int*)malloc(sizeof(unsigned int));
    enc1[0] = encoded_X_test[0];
    int* y_p1 = (int*)malloc(sizeof(int));
    long t_m_s = currentMillis();
    itm_predict(itm, enc1, y_p1, 1);

    long t_m_e = currentMillis();
    printf("Time to run one test: %f\n", (t_m_e-t_m_s)/1000.0);
    free(y_p1);
    free(enc1);
    
    itm_predict(itm, encoded_X_test, y_pred, num_test_data);

    int num_correct = 0;
    int num_wrong = 0;
    for (int i=0; i<num_test_data; ++i) {
        if (y_test[i] == y_pred[i]) {
            num_correct += 1;
        } else {
            num_wrong += 1;
        }
    }
    printf("Num correct: %d, Num wrong: %d\n", num_correct, num_wrong);
    printf("Accuracy: %.2f%%\n", 100*(num_correct)/(1.0*(num_correct + num_wrong)));

    mc_tm_destroy(itm->mc_tm);
    free(itm->mc_tm);
    itm_destroy(itm);
    return 0;
}

