#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void read_samples_file(FILE* fp, int num_samples, int num_features, unsigned int* X_data, int* y_data)
{
	char * line = NULL;
	size_t len = 0;

	const char *s = " ";
	char *token = NULL;

	for (int i = 0; i < num_samples; i++) {
		getline(&line, &len, fp);

		token = strtok(line, s);
		for (int j = 0; j < num_features; j++) {
			X_data[i*num_features + j] = atoi(token);
			token=strtok(NULL,s);
		}
		y_data[i] = atoi(token);
	}
	free(line);
}

