#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "Tools.h"
#include "IndexedTsetlinMachine.h"
#include "utils.h"
#include "train_and_output_model.h"

void print_model_to_CSV(char* filename, struct ModelInputs model_inputs,
                        struct OtherModelInputs omi,
                        struct IndexedTsetlinMachine* itm, int append_negated) {
    FILE * fp;
    fp = fopen(filename, "w");

    // Inputs needed for MCTM
    fprintf(fp, "Model Inputs\n");
    fprintf(fp, "number_of_classes\n");
    fprintf(fp, "%d\n", model_inputs.number_of_classes);

    fprintf(fp, "number_of_clauses\n");
    fprintf(fp, "%d\n", model_inputs.number_of_clauses);

    fprintf(fp, "number_of_features\n");
    fprintf(fp, "%d\n", model_inputs.number_of_features);

    fprintf(fp, "number_of_patches\n");
    fprintf(fp, "%d\n", omi.number_of_patches);

    fprintf(fp, "number_of_ta_chunks\n");
    fprintf(fp, "%d\n", omi.number_of_ta_chunks);

    fprintf(fp, "number_of_state_bits\n");
    fprintf(fp, "%d\n", omi.number_of_state_bits);

    fprintf(fp, "T_val\n");
    fprintf(fp, "%d\n", model_inputs.T_val);
    
    fprintf(fp, "s_val\n");
    fprintf(fp, "%f\n", model_inputs.s_val);

    fprintf(fp, "s_range\n");
    fprintf(fp, "%f\n", omi.s_range);

    fprintf(fp, "boost_true_positive_feedback\n");
    fprintf(fp, "%d\n", omi.boost_true_positive_feedback);

    fprintf(fp, "weighted_clauses\n");
    fprintf(fp, "%d\n", omi.weighted_clauses);

    fprintf(fp, "clause_drop_p\n");
    fprintf(fp, "%f\n", omi.clause_drop_p);
    
    fprintf(fp, "literal_drop_p\n");
    fprintf(fp, "%f\n", omi.literal_drop_p);

    fprintf(fp, "max_included_literals\n");
    fprintf(fp, "%d\n", omi.max_included_literals); 

    fprintf(fp, "append_negated\n");
    fprintf(fp, "%d\n", append_negated);  

    fprintf(fp, "total_num_features\n");
    // append_negated doubles the features, is either 0 or 1
    fprintf(fp, "%d\n", model_inputs.number_of_features*(1 + append_negated));

    // Loop through the TMs
    fprintf(fp, "\nTsetlinMachine List\n");
    int num_tms = itm->mc_tm->number_of_classes;
    fprintf(fp, "%d\n", num_tms);

    for (int i=0; i<num_tms; ++i) {
        struct TsetlinMachine* tm = itm->mc_tm->tsetlin_machines[i];
        fprintf(fp, "\nTsetlin Machine %d\n", i);

        fprintf(fp, "ta_state\n");
        int num_ta_state = tm->number_of_clauses * tm->number_of_ta_chunks * tm->number_of_state_bits;
        fprintf(fp, "%d\n", num_ta_state);
        for (int j=0; j<num_ta_state; ++j) {
            if ((j != 0) & ((j % 10000) == 0)) {
                fprintf(fp, "\n");
            }

            fprintf(fp, "%u,", tm->ta_state[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "clause_output\n");
        int num_clause_output = tm->number_of_clause_chunks;
        fprintf(fp, "%d\n", num_clause_output);
        for (int j=0; j<num_clause_output; ++j) {
            fprintf(fp, "%u,", tm->clause_output[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "drop_clause\n");
        int num_drop_clause = tm->number_of_clause_chunks;
        fprintf(fp, "%d\n", num_drop_clause);
        for (int j=0; j<num_drop_clause; ++j) {
            fprintf(fp, "%u,", tm->drop_clause[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "drop_literal\n");
        int num_drop_literal = tm->number_of_ta_chunks;
        fprintf(fp, "%d\n", num_drop_literal);
        for (int j=0; j<num_drop_literal; ++j) {
            fprintf(fp, "%u,", tm->drop_literal[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "feedback_to_la\n");
        int num_feedback_to_la = tm->number_of_ta_chunks;
        fprintf(fp, "%d\n", num_feedback_to_la);
        for (int j=0; j<num_feedback_to_la; ++j) {
            fprintf(fp, "%u,", tm->feedback_to_la[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "feedback_to_clauses\n");
        int num_feedback_to_clauses = tm->number_of_clause_chunks;
        fprintf(fp, "%d\n", num_feedback_to_clauses);
        for (int j=0; j<num_feedback_to_clauses; ++j) {
            fprintf(fp, "%d,", tm->feedback_to_clauses[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "clause_patch\n");
        int num_clause_patch = tm->number_of_clauses;
        fprintf(fp, "%d\n", num_clause_patch);
        for (int j=0; j<num_clause_patch; ++j) {
            fprintf(fp, "%u,", tm->clause_patch[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "output_one_patches\n");
        int num_output_one_patches = tm->number_of_patches;
        fprintf(fp, "%d\n", num_output_one_patches);
        for (int j=0; j<num_output_one_patches; ++j) {
            fprintf(fp, "%d,", tm->output_one_patches[j]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "clause_weights\n");
        int num_clause_weights = tm->number_of_clauses;
        fprintf(fp, "%d\n", num_clause_weights);
        for (int j=0; j<num_clause_weights; ++j) {
            fprintf(fp, "%u,", tm->clause_weights[j]);
        }
        fprintf(fp, "\n");
    }


    // IndexedTsetlinMachine
    fprintf(fp, "\nIndexedTsetlinMachine\n");
    fprintf(fp, "clause_state\n");
    int num_clause_state = itm->mc_tm->number_of_classes * itm->mc_tm->tsetlin_machines[0]->number_of_clauses * itm->mc_tm->tsetlin_machines[0]->number_of_features;
    fprintf(fp, "%d\n", num_clause_state);
    for (int i=0; i<num_clause_state; ++i) {
        fprintf(fp, "%u,", itm->clause_state[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "baseline_class_sum\n");
    int num_bcs = itm->mc_tm->number_of_classes;
    fprintf(fp, "%d\n", num_bcs);
    for (int i=0; i<num_bcs; ++i) {
        fprintf(fp, "%d,", itm->baseline_class_sum[i]);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "class_feature_list\n");
    int num_cfl = itm->mc_tm->number_of_classes * (itm->mc_tm->tsetlin_machines[0]->number_of_clauses + 1) * itm->mc_tm->tsetlin_machines[0]->number_of_features * 2;
    fprintf(fp, "%d\n", num_cfl);
    for (int i=0; i<num_cfl; ++i) {
        fprintf(fp, "%d,", itm->class_feature_list[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "class_feature_pos\n");
    int num_cfp = itm->mc_tm->number_of_classes * itm->mc_tm->tsetlin_machines[0]->number_of_clauses * itm->mc_tm->tsetlin_machines[0]->number_of_features;
    fprintf(fp, "%d\n", num_cfp);
    for (int i=0; i<num_cfp; ++i) {
        fprintf(fp, "%d,", itm->class_feature_pos[i]);
    }
    fprintf(fp, "\n");

    fclose(fp);
}

// train_and_output_model_to_CSV - takes in model inputs and training data
// X_train is an unsigned int* which is an already featurized and flattened input of length (#samples)*(#features)
// y_train is an int* of length (#samples)
void train_and_output_model_to_CSV(char* filename, struct ModelInputs model_inputs, int num_samples, unsigned int* X_train, int* y_train) {

    // Features mult by 2 if using negated which we are
    // Can add that later as a choice
    int append_negated = 1;
    int total_features = model_inputs.number_of_features*(1+append_negated);

    struct OtherModelInputs omi = {
        .append_negated = append_negated,
        .total_features = total_features,
    
        // Can leave these as is for now
        // some of them should later be allowed to be updated
        // all should be written to the file for use
        .number_of_patches = 1,
        .number_of_ta_chunks = (int)(total_features-1)/32 + 1,
        .number_of_state_bits = 8,  
        .s_range = model_inputs.s_val,
        .boost_true_positive_feedback = 1, // should be default 1;
        .weighted_clauses = 0, // false
        .clause_drop_p = 0.0,
        .literal_drop_p = 0.0,
        .max_included_literals = total_features
    };

    struct MultiClassTsetlinMachine* mtm;
    mtm = CreateMultiClassTsetlinMachine(
        model_inputs.number_of_classes,
        model_inputs.number_of_clauses,
        total_features,
        omi.number_of_patches,
        omi.number_of_ta_chunks,
        omi.number_of_state_bits,
        model_inputs.T_val,
        model_inputs.s_val,
        omi.s_range,
        omi.boost_true_positive_feedback,
        omi.weighted_clauses,
        omi.clause_drop_p,
        omi.literal_drop_p,
        omi.max_included_literals
    );

    mc_tm_initialize(mtm);

    int num_ta_chunks = (int) (2*model_inputs.number_of_features - 1)/32 + 1;
    int len_encoded_x_train = num_samples*num_ta_chunks;
    unsigned int* encoded_X_train = (unsigned int*)malloc(len_encoded_x_train*sizeof(unsigned int));
    tm_encode(X_train, encoded_X_train, num_samples, model_inputs.number_of_features, 1, 1, model_inputs.number_of_features, 1, append_negated, 0);
    
    FILE* fp_enc = fopen("data/x_train_enc.txt", "w");
    for (int i=0; i<len_encoded_x_train; ++i) {
        fprintf(fp_enc, "%u\n", encoded_X_train[i]);
    }
    fclose(fp_enc);

    struct IndexedTsetlinMachine* itm = CreateIndexedTsetlinMachine(mtm);
    itm_fit(itm, encoded_X_train, y_train, num_samples, model_inputs.number_of_epochs);

    // NOW OUTPUT FILE
    // Each param and builtin stuff
    print_model_to_CSV(filename, model_inputs, omi, itm, append_negated);

    mc_tm_destroy(itm->mc_tm);
    free(itm->mc_tm);
    itm_destroy(itm);
    free(encoded_X_train);
}

int main(int argc, char* argv[]) {
    time_t time_start = time(NULL);

    if (argc > 2) {
        return 1;
    }

    char* input_filename;
    if (argc == 1) {
        input_filename = "data/txt_cpp_features_training.txt";
        printf("Running default CPP version: %s\n", input_filename);
    } else if (strcmp(argv[1], "delta") == 0) {
        input_filename = "data/txt_cpp_features_training_delt.txt";
        printf("Running delta CPP version: %s\n", input_filename);
    } else if (strcmp(argv[1], "py") == 0) {
        input_filename = "data/txt_py_train_output_feat.txt";
        printf("Running Python version: %s\n", input_filename); 
    } else if (strcmp(argv[1], "c") == 0) {
        input_filename = "data/txt_c_train_features.txt";
        printf("Running C version: %s\n", input_filename);
    } else if (strcmp(argv[1], "xor") == 0) {
        input_filename = "data/NoisyXORTrainingData.txt";
        printf("Running XOR version: %s\n", input_filename);
    } else {
        return 1;
    }

    FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(input_filename, "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

    getline(&line, &len, fp);
    int num_training = atoi(line);
    getline(&line, &len, fp);
    int num_features = atoi(line);
    getline(&line, &len, fp);
    int num_classes = atoi(line);
    free(line);

    printf("training: %d, features: %d, Classes: %d\n", num_training, num_features, num_classes);

    struct ModelInputs model_inputs = {
        .number_of_classes = num_classes,
        .number_of_clauses = 400,
        .T_val = 50,
        .s_val = 4.0,
        .number_of_features = num_features,
        .number_of_epochs = 200,
    };    
    
    char filename[50] = "data/txt_model_output.txt";

    int num_samples = num_training;
    unsigned int* X_train = (unsigned int*)malloc(num_samples*model_inputs.number_of_features*sizeof(unsigned int));
    int* y_train = (int*)malloc(num_samples*sizeof(int));

    read_samples_file(fp, num_samples, model_inputs.number_of_features, X_train, y_train);

    time_t time_readfile = time(NULL);
    printf("Time elapsed read in file: %ld\n", time_readfile - time_start);

    train_and_output_model_to_CSV(filename, model_inputs, num_samples, X_train, y_train);

    free(X_train);
    free(y_train);

    time_t time_trainmodel = time(NULL);
    printf("Time elapsed train model: %ld\n", time_trainmodel - time_start);

    return 0;
}

