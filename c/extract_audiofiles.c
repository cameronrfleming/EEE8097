#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "extract_audiofiles.h"

const int NUM_AUDIO_FILES = 400;

struct filedata* filter_audio_files(char *meta_data_path) {
    FILE* fp = fopen(meta_data_path, "r");
 
    if (!fp) {
        printf("Can't open file\n");
        return NULL;
    }
    
    // Buffer for reading in values
    char buffer[1024];

    int row = 0;
    int column = 0;
    
    // Store in a 2D char array
    // total of 400 rows we will end up with
    // Columns are as follows:
    // filename (string), fold (int), target (int), category (string), esc10 (bool) - filtering on this,
    // src_file (int?), take (char)
    // filename is a concatenation of fold + src_file + take + target + ".wav"
    char** audio_files = (char**)malloc(NUM_AUDIO_FILES * sizeof(char*));
    if (audio_files == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    char** categories = (char**)malloc(NUM_AUDIO_FILES * sizeof(char*));

    int afl_ctr = 0;
    const int max_filename_size = 32;

    while (fgets(buffer,1024, fp)) {
        column = 0;
        row++;

        // Row 1 is the title of the columns
        if (row == 1)
            continue;

        // Splitting the data
        char* value = strtok(buffer, ", ");

        char* filename;
        char* category;

        while (value) {
            // Column 1 - filename
            if (column == 0) {
                filename = value;
            }

            // Column 4 - category
            if (column == 3) {
                category = value;
            }

            // Column 5
            if (column == 4) {
                // if val is true, put filename in list and increment afl_ctr
                if (strcmp(value, "True") == 0) {
                    audio_files[afl_ctr] = (char*)malloc(max_filename_size * sizeof(char));
                    if (audio_files[afl_ctr] == NULL) {
                        fprintf(stderr, "Memory allocation failed\n");
                        return NULL;
                    }
                    sprintf(audio_files[afl_ctr], "%s", filename);
                    categories[afl_ctr] = (char*)malloc(100*sizeof(char));
                    sprintf(categories[afl_ctr], "%s", category);
                    afl_ctr++;
                }
            }
            value = strtok(NULL, ", ");
            column++;
        }
    }
    fclose(fp);
    struct filedata* fd = (struct filedata*)malloc(sizeof(struct filedata));
    fd->audio_files = audio_files;
    fd->categories = categories;
    return fd;
}
