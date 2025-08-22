#ifndef BUILD_MODEL_FROM_CSV_H
#define BUILD_MODEL_FROM_CSV_H

struct IndexedTsetlinMachine* build_indexedTM_model_from_CSV(char* filename, int* append_negated);

struct IndexedTsetlinMachine* build_itm_from_file_and_test(char* model_filename, char* testdata_filename, int num_samples);

#endif
