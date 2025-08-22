CC = gcc

# C feature extraction
lpreprocess:
	$(CC) -c extract_audiofiles.c
	$(CC) -c librosa.c
	$(CC) -c filters.c
	$(CC) -I/usr/local/opt/fftw/include -c mfcc.c
	$(CC) -c preprocess_data.c
	$(CC) -c binarize.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c EmbeddingTsetlinMachine.c 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c MultiClassConvolutionalTsetlinMachine.c 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c ConvolutionalTsetlinMachine.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c Tools.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c IndexedTsetlinMachine.c	
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c utils.c	
	$(CC) -I/usr/local/opt/fftw/include preprocessing.c -o preprocessing -lsndfile -lfftw3f extract_audiofiles.o librosa.o filters.o mfcc.o preprocess_data.o binarize.o EmbeddingTsetlinMachine.o MultiClassConvolutionalTsetlinMachine.o ConvolutionalTsetlinMachine.o Tools.o IndexedTsetlinMachine.o utils.o

# Train and print model to file
ltrainmodel:
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c EmbeddingTsetlinMachine.c 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c MultiClassConvolutionalTsetlinMachine.c 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c ConvolutionalTsetlinMachine.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c Tools.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c IndexedTsetlinMachine.c	
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c utils.c	
	$(CC) train_and_output_model.c -o train_and_output_model EmbeddingTsetlinMachine.o MultiClassConvolutionalTsetlinMachine.o ConvolutionalTsetlinMachine.o Tools.o IndexedTsetlinMachine.o utils.o -lm

# Build model from CSV and run testing
lbuildfromcsv:
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c EmbeddingTsetlinMachine.c 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c MultiClassConvolutionalTsetlinMachine.c 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c ConvolutionalTsetlinMachine.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c Tools.c
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c IndexedTsetlinMachine.c	
	gcc -O3 -ffast-math -c -Wall -Werror -fpic -c utils.c
	gcc -c build_model_from_csv.c		
	$(CC) -I/usr/local/opt/fftw/include -c mfcc.c
	$(CC) -I/usr/local/opt/fftw/include test_model_from_csv.c -o test_model_from_csv EmbeddingTsetlinMachine.o MultiClassConvolutionalTsetlinMachine.o ConvolutionalTsetlinMachine.o Tools.o IndexedTsetlinMachine.o utils.o build_model_from_csv.o -lfftw3f

