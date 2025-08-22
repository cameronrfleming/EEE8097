# EEE8097
Low-cost Environmental Sound Classification using Tsetlin Machines - C, C++, and Python Implementations

To run the python version:
1. Build a virtual environment: `python3 -m venv venv`
2. activate it: `source venv/bin/activate`
3. pip install: pyTsetlinMachine, scikit-learn, numpy, librosa, pandas, tqdm, resampy
4. Run: `python performance_testing.py`

Libraries needed for preprocessing in C:

Run `sudo apt-get install <lib>`
* libfftw3-dev
* libsndfile1-dev

Libraries needed for preprocessing in C++:
* wavreader.c wavreader.h (https://sources.debian.org/src/vo-amrwbenc/0.1.2-1/wavreader.c/)
* FiltFilt (https://github.com/KBaur/FiltFilt)
* LibrosaCpp (https://github.com/ewan-xu/LibrosaCpp)

To preprocess data using C:
1. `cd c`
2. `make lpreprocess`
3. `./preprocessing`

To preprocess data using C++:
1. `cd cpp`
2. `g++ feature_extraction.cpp wavreader.c`
3. `./a.out`
4. `cp cpp_features_training.txt ../c/data/txt_cpp_features_training.txt`
5. `cp cpp_features_test.txt ../c/data/txt_cpp_features_test.txt`

To train the model (in C):
1. `cd c`
2. `make ltrainmodel`
3. For C: `./train_and_output_model c`, for C++: `./train_and_output_model`

To run inference on the model (in C):
1. `cd c`
2. `make lbuildfromcsv`
3. For C: `./test_model_from_csv c`, for C++: `./test_model_from_csv`
