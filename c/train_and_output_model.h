#ifndef TRAIN_AND_OUTPUT_MODEL
#define TRAIN_AND_OUTPUT_MODEL

struct ModelInputs {
    int number_of_classes;
    int number_of_clauses;
    int T_val;
    double s_val;
    int number_of_features;
    int number_of_epochs;
};

// Some of these have specific values, some can be changed
struct OtherModelInputs {
    int append_negated;
    int total_features;
    int number_of_patches;
    int number_of_ta_chunks;
    int number_of_state_bits;    
    double s_range; // gets set to s (usually)
    int boost_true_positive_feedback; // should be default 1;
    int weighted_clauses;
    float clause_drop_p;
    float literal_drop_p;
    int max_included_literals;
};

void train_and_output_model_to_CSV(char* filename, struct ModelInputs model_inputs, int num_samples, unsigned int* X_train, int* y_train);

#endif

