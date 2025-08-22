#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "IndexedTsetlinMachine.h"
#include "utils.h"
#include "Tools.h"

struct IndexedTsetlinMachine* build_indexedTM_model_from_CSV(char* filename, int* append_negated) {
    FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

    // create the MCTM first
    // then set all the values for each of it's TMs
    // then build the ITM from it and set it's values

    // Read in MultiClassTsetlinMachine
    getline(&line, &len, fp); // Model Inputs
    getline(&line, &len, fp); // number_of_classes
    getline(&line, &len, fp); // int
    int number_of_classes = atoi(line);

    getline(&line, &len, fp); // number_of_clauses
    getline(&line, &len, fp); // int
    int number_of_clauses = atoi(line);

    getline(&line, &len, fp); // number_of_features
    getline(&line, &len, fp); // int
    int number_of_features = atoi(line);

    getline(&line, &len, fp); // number_of_patches
    getline(&line, &len, fp); // int
    int number_of_patches = atoi(line);

    getline(&line, &len, fp); // number_of_ta_chunks
    getline(&line, &len, fp); // int
    int number_of_ta_chunks = atoi(line);

    getline(&line, &len, fp); // number_of_state_bits
    getline(&line, &len, fp); // int
    int number_of_state_bits = atoi(line);
    
    getline(&line, &len, fp); // T_val
    getline(&line, &len, fp); // int
    int T_val = atoi(line);
    
    getline(&line, &len, fp); // s_val
    getline(&line, &len, fp); // double
    double s_val = atof(line);
    
    getline(&line, &len, fp); // s_range
    getline(&line, &len, fp); // double
    double s_range = atof(line);
    
    getline(&line, &len, fp); // boost_true_positive_feedback
    getline(&line, &len, fp); // int
    int boost_true_positive_feedback = atoi(line);

    getline(&line, &len, fp); // weighted_clauses
    getline(&line, &len, fp); // int
    int weighted_clauses = atoi(line);

    getline(&line, &len, fp); // clause_drop_p
    getline(&line, &len, fp); // double 
    double clause_drop_p = atof(line);

    getline(&line, &len, fp); // literal_drop_p
    getline(&line, &len, fp); // double 
    double literal_drop_p = atof(line);

    getline(&line, &len, fp); // max_included_literals
    getline(&line, &len, fp); // int
    int max_included_literals = atoi(line);

    getline(&line, &len, fp); // append_negated
    getline(&line, &len, fp); // int
    (*append_negated) = atoi(line);

    getline(&line, &len, fp); // total_num_features
    getline(&line, &len, fp); // int
    int total_num_features = atoi(line);

    struct MultiClassTsetlinMachine* mc_tm = CreateMultiClassTsetlinMachine(
        number_of_classes,
        number_of_clauses,
        total_num_features,
        number_of_patches,
	    number_of_ta_chunks,
	    number_of_state_bits,
	    T_val,
	    s_val,
	    s_range,
	    boost_true_positive_feedback,
	    weighted_clauses,
	    clause_drop_p,
	    literal_drop_p,
	    max_included_literals
    );
    // MCTM is now fully built, let's set the stuff for the TMs
    
    // TsetlinMachine List
    getline(&line, &len, fp); // empty line
    getline(&line, &len, fp); // TsetlinMachine List
    getline(&line, &len, fp); // num tsetlin machines (num classes)
    int num_tms = atoi(line);
    getline(&line, &len, fp); // empty line

    // separating things with a comma here
	const char *s = ",";
	char *token = NULL;
    
    // Loop through the TMs
    for (int i=0; i<num_tms; ++i) {
        getline(&line, &len, fp); // #tm

        // TA states are huge so only have 10k per line to prevent
        // memory issues when reading lines in
        // ta_state
        getline(&line, &len, fp); // ta_state
        getline(&line, &len, fp); // int
        int num_ta_state = atoi(line);
        int countTA = 0;
        const int maxTAline = 10000;
        while (countTA < num_ta_state) {
            getline(&line, &len, fp);
            token = strtok(line, s);
            if (countTA + maxTAline < num_ta_state) {
                for (int j=0; j<maxTAline; ++j) {
                    char* endptr;
                    unsigned int ta_state = strtoul(token, &endptr, 0);
                    mc_tm->tsetlin_machines[i]->ta_state[countTA + j] = ta_state;
                    token=strtok(NULL,s);
                } 
            } else {
                for (int j=0; j<num_ta_state-countTA; ++j) {
                    char* endptr;
                    unsigned int ta_state = strtoul(token, &endptr, 0);
                    mc_tm->tsetlin_machines[i]->ta_state[countTA + j] = ta_state;
                    token=strtok(NULL,s);
                }  
            }
            countTA += maxTAline;
        }

        // clause_output
        getline(&line, &len, fp); // clause_output
        getline(&line, &len, fp); // int
        int num_clause_output = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop through clause output 
        for (int j=0; j<num_clause_output; ++j) {
            char* endptr;
            unsigned int clause_output = strtoul(token, &endptr, 0);
            mc_tm->tsetlin_machines[i]->clause_output[j] = clause_output;
			token=strtok(NULL,s);
        }

        // drop_clause
        getline(&line, &len, fp); // drop_clause
        getline(&line, &len, fp); // int
        int num_drop_clause = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop through drop clause
        for (int j=0; j<num_drop_clause; ++j) {
            char* endptr;
            unsigned int drop_clause = strtoul(token, &endptr, 0);
            mc_tm->tsetlin_machines[i]->drop_clause[j] = drop_clause;
			token=strtok(NULL,s);
        }

        // drop_literal
        getline(&line, &len, fp); // drop_literal
        getline(&line, &len, fp); // int
        int num_drop_literal = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop through drop literal
        for (int j=0; j<num_drop_literal; ++j) {
            char* endptr;
            unsigned int drop_literal = strtoul(token, &endptr, 0);
            mc_tm->tsetlin_machines[i]->drop_literal[j] = drop_literal;
			token=strtok(NULL,s);
        }

        // feedback_to_la
        getline(&line, &len, fp); // feedback_to_la
        getline(&line, &len, fp); // int
        int num_feedback_to_la = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop through 
        for (int j=0; j<num_feedback_to_la; ++j) {
            char* endptr;
            unsigned int feedback_to_la = strtoul(token, &endptr, 0);
            mc_tm->tsetlin_machines[i]->feedback_to_la[j] = feedback_to_la;
			token=strtok(NULL,s);
        }

        // feedback_to_clauses
        getline(&line, &len, fp); // feedback_to_clauses
        getline(&line, &len, fp); // int
        int num_feedback_to_clauses = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop 
        for (int j=0; j<num_feedback_to_clauses; ++j) {
            int feedback_to_clauses = atoi(token);
            mc_tm->tsetlin_machines[i]->feedback_to_clauses[j] = feedback_to_clauses;
			token=strtok(NULL,s);
        }

        // clause_patch
        getline(&line, &len, fp); // clause_patch
        getline(&line, &len, fp); // int
        int num_clause_patch = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop
        for (int j=0; j<num_clause_patch; ++j) {
            char* endptr;
            unsigned int clause_patch = strtoul(token, &endptr, 0);
            mc_tm->tsetlin_machines[i]->clause_patch[j] = clause_patch;
			token=strtok(NULL,s);
        }

        // output_one_patches
        getline(&line, &len, fp); // output_one_patches
        getline(&line, &len, fp); // int
        int num_output_one_patches = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop 
        for (int j=0; j<num_output_one_patches; ++j) {
            int output_one_patches = atoi(token);
            mc_tm->tsetlin_machines[i]->output_one_patches[j] = output_one_patches;
			token=strtok(NULL,s);
        }

        // clause_weights
        getline(&line, &len, fp); // clause_weights
        getline(&line, &len, fp); // int
        int num_clause_weights = atoi(line);

        getline(&line, &len, fp);
		token = strtok(line, s);
        // Loop
        for (int j=0; j<num_clause_weights; ++j) {
            char* endptr;
            unsigned int clause_weights = strtoul(token, &endptr, 0);
            mc_tm->tsetlin_machines[i]->clause_weights[j] = clause_weights;
			token=strtok(NULL,s);
        }
        getline(&line, &len, fp); // empty line
    }

    // TMs Set up!

    // Now ITM
    struct IndexedTsetlinMachine* itm = CreateIndexedTsetlinMachine(mc_tm);
    
    // getline(&line, &len, fp); // empty
    getline(&line, &len, fp); // IndexedTsetlinMachine

    // clause_state
    getline(&line, &len, fp); // clause_state
    getline(&line, &len, fp); // int
    int num_clause_state = atoi(line);

    getline(&line, &len, fp);
    token = strtok(line, s);
    // Loop
    for (int i=0; i<num_clause_state; ++i) {
        char* endptr;
        unsigned int clause_state = strtoul(token, &endptr, 0);
        itm->clause_state[i] = clause_state;
        token=strtok(NULL,s);
    }

    // baseline_class_sum
    getline(&line, &len, fp); // baseline_class_sum
    getline(&line, &len, fp); // int
    int num_baseline_class_sum = atoi(line);

    getline(&line, &len, fp);
    token = strtok(line, s);
    // Loop
    for (int i=0; i<num_baseline_class_sum; ++i) {
        // char* endptr;
        unsigned int baseline_class_sum = atoi(token);
        itm->baseline_class_sum[i] = baseline_class_sum;
        token=strtok(NULL,s);
    }

    // class_feature_list
    getline(&line, &len, fp); // class_feature_list
    getline(&line, &len, fp); // int
    int num_class_feature_list = atoi(line);

    getline(&line, &len, fp);
    token = strtok(line, s);
    // Loop
    for (int i=0; i<num_class_feature_list; ++i) {
        // char* endptr;
        unsigned int class_feature_list = atoi(token);
        itm->class_feature_list[i] = class_feature_list;
        token=strtok(NULL,s);
    }

    // class_feature_pos
    getline(&line, &len, fp); // class_feature_pos
    getline(&line, &len, fp); // int
    int num_class_feature_pos = atoi(line);

    getline(&line, &len, fp);
    token = strtok(line, s);
    // Loop
    for (int i=0; i<num_class_feature_pos; ++i) {
        unsigned int class_feature_pos = atoi(token);
        itm->class_feature_pos[i] = class_feature_pos;
        token=strtok(NULL,s);
    }

    free(line);
    fclose(fp);

    return itm;
}
