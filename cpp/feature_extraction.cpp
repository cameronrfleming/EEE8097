#include <iostream>
using std::cout;
using std::endl;
using std::cerr;

#include <fstream>
#include <iomanip>
#include <chrono>

#include "librosa/librosa.h"
#include "wavreader.h"
#include "FiltFilt.h"

struct split_data {
    std::vector<std::string> training_filenames;
    std::vector<int> training_categories;
    std::vector<std::string> test_filenames;
    std::vector<int> test_categories;
};

std::vector<std::vector<float>> get_binarization_thresholds(std::vector<std::vector<float>> x_train, int resolution) {
    int num_samples = x_train.size();
    int num_features = x_train[0].size();
    float feature_mins[num_features];
    float feature_maxs[num_features];

    for (int i=0; i<num_features; ++i) {
        feature_mins[i] = x_train[0][i];
        feature_maxs[i] = x_train[0][i];
    }
    for (int i=1; i<num_samples; ++i) {
        for (int j=0; j<num_features; ++j) {
            if (x_train[i][j] < feature_mins[j]) {
                feature_mins[j] = x_train[i][j];
            }
            if (x_train[i][j] > feature_maxs[j]) {
                feature_maxs[j] = x_train[i][j];
            }
        }
    }

    // Loop through and set the thresholds
    std::vector<std::vector<float>> thresholds(num_features, std::vector<float>(resolution, 0));
    for (int j=0; j<num_features; ++j) {
        for (int k=0; k<resolution; ++k) {
            float step_size = (feature_maxs[j] - feature_mins[j])/(1.0f * resolution);
            thresholds[j][k] = feature_mins[j] + k*step_size;
        }
    }
    return thresholds;
}

std::vector<std::vector<int8_t>> binarize_features(std::vector<std::vector<float>> X, std::vector<std::vector<float>> thresholds, int res) {
    int num_samples = X.size();
    int num_features = X[0].size();
    std::vector<std::vector<int8_t>> X_binarized(num_samples, std::vector<int8_t>(num_features*res));
    for (int i=0; i<num_samples; ++i) {
        for (int j=0; j<num_features; ++j) {
            for (int k=0; k<res; ++k) {
                if (X[i][j] > thresholds[j][k]) {
                    X_binarized[i][j*res + k] = 1;
                } else {
                    X_binarized[i][j*res + k] = 0;
                }
            }
        }
    }
    return X_binarized;
}

std::vector<std::vector<float>> convolve1d(std::vector<std::vector<float>> mfccs, std::vector<float> window) {
    // 1-D so does it to each row
    // flip the window
    int window_size = window.size();
    std::vector<float> flipwindow(window_size);
    for (int i=0; i<window_size; ++i) {
        flipwindow[i] = window[window_size-i-1];
    }

    std::vector<std::vector<float>> convolved(mfccs.size(), std::vector<float>(mfccs[0].size()));
    int mid_point = (int) window_size/2;
    for (int i=0; i<mfccs[0].size(); ++i) {
        for (int j=0; j<mfccs.size(); ++j) {
            float sum = 0.0;
            for (int k=0; k<window_size; ++k) {
                int loc = j + k - mid_point;
                if (loc >= 0 & loc < mfccs.size()) {
                    sum += mfccs[loc][i] * flipwindow[k];
                }
            }
            convolved[j][i] = sum;
        }
    }
    return convolved; 
}

std::vector<float> read_file_and_preprocess(std::string filename) {
    std::stringstream ss;
    ss << "../c_implementation/tsetlin/env_data/ESC-50-master/audio/" << filename;
    void* h_x = wav_read_open(ss.str().c_str());
    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res)
    {
        cerr << "get ref header error: " << res << endl;
    }

    int samples = data_length * 8 / bits_per_sample;
    std::vector<int16_t> tmp(samples);
    res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
    if (res < 0)
    {
        cerr << "read wav file error: " << res << endl;
    }
    std::vector<float> x(samples);
    std::transform(tmp.begin(), tmp.end(), x.begin(),
        [](int16_t a) {
        // this division is to turn an int into a float between -1 and 1?
        return static_cast<float>(a) / 32767.f;
    });
    
    // Need to send through low and high pass filter first
    // Get coeffs from matlab
    //https://github.com/KBaur/FiltFilt

    // Low pass
    kb::math::FilterCoefficients<float> fcl{ 
        m_CoefficientsA:{1.0000, -1.21887934, 0.44768083}, 
        m_CoefficientsB:{0.05720037, 0.11440075, 0.05720037}
    };
    kb::math::FiltFilt<float> filtfiltL(fcl);
    auto lowPassed = filtfiltL.ZeroPhaseFiltering(x);

    kb::math::FilterCoefficients<float> fch{ 
        m_CoefficientsA:{1.0000, -1.93957021, 0.9413433}, 
        m_CoefficientsB:{0.97022838, -1.94045675, 0.97022838}
    };
    kb::math::FiltFilt<float> filtfiltH(fch);
    auto highPassed = filtfiltH.ZeroPhaseFiltering(lowPassed);

    // MFCCs time
    int n_fft = 2048;
    int n_hop = 512;
    int n_mel_mfcc = 128;
    int n_mfcc = 13;
    int fmin = 0;
    int fmax = sr/2;
    bool norm = true;
    int type = 2;
    auto mfccs = librosa::Feature::mfcc(highPassed, sr, n_fft, n_hop, "hann", true, "reflect", 2.f, n_mel_mfcc, fmin, fmax, n_mfcc, norm, type);

    // DELTA MFCCs
    std::vector<float> d_coeffs = { 6.66666667e-02,  5.00000000e-02,  3.33333333e-02,  1.66666667e-02,
        -3.46944695e-18, -1.66666667e-02, -3.33333333e-02, -5.00000000e-02,
        -6.66666667e-02};
    std::vector<float> dd_coeffs = { 0.06060606,  0.01515152, -0.01731602, -0.03679654, -0.04329004, -0.03679654,
        -0.01731602,  0.01515152,  0.06060606};

    std::vector<std::vector<float>> d_mfccs = convolve1d(mfccs, d_coeffs);
    std::vector<std::vector<float>> dd_mfccs = convolve1d(mfccs, dd_coeffs);

    std::vector<float> avg_mfccs(n_mfcc*3);
    int num_frames_mfcc = mfccs.size();
    for (int i=0; i<n_mfcc; ++i) {
        float sum = 0.0;
        float sum_d = 0.0;
        float sum_dd = 0.0;
        for (int j=0; j<num_frames_mfcc; ++j) {
            sum += mfccs[j][i];
            sum_d += d_mfccs[j][i];
            sum_dd += dd_mfccs[j][i];
        }
        avg_mfccs[i] = sum/(1.0 * num_frames_mfcc);
        avg_mfccs[i+n_mfcc] = sum_d/(1.0 * num_frames_mfcc);
        avg_mfccs[i+ 2*n_mfcc] = sum_dd/(1.0 * num_frames_mfcc);
    }

    // melspectrogram
    int n_mels_spec = 64;
    auto melspectrogram = librosa::Feature::melspectrogram(highPassed, sr, n_fft, n_hop, "hann", true, "reflect", 2.f, n_mels_spec, fmin, fmax);

    float mel_scaled[n_mels_spec];
    int num_frames_ms = melspectrogram.size();
    for (int i=0; i<n_mels_spec; ++i) {
        float sum = 0.0;
        for (int j=0; j<num_frames_ms; ++j) {
            sum += melspectrogram[j][i];
        }
        mel_scaled[i] = sum/(1.0 * num_frames_ms);
    }

    // vector for the features
    std::vector<float> feature_vec(n_mels_spec + avg_mfccs.size());
    for (int i=0; i<n_mels_spec; ++i) {
        feature_vec[i] = mel_scaled[i];
    }
    for (int i=0; i<avg_mfccs.size(); ++i) {
        feature_vec[n_mels_spec + i] = avg_mfccs[i];
    }
    return feature_vec;
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

std::vector<int> encode_labels(std::vector<std::string> categories, int num_classes) {
    int num_audio_files = categories.size();
    std::vector<std::string> labels(num_classes);
    std::vector<int> encoded_labels(categories.size());
    int label_loc = 0;

    for (int i=0; i<num_audio_files; ++i) {
        if (label_loc < num_classes) {
            int found = 0;
            for (int j=0; j<label_loc; ++j) {
                if (labels[j].compare(categories[i]) == 0) {
                    encoded_labels[i] = j;
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                labels[label_loc] = categories[i];
                encoded_labels[i] = label_loc;
                label_loc += 1;
            }
        } else {
            int found = 0;
            for (int j=0; j<num_classes; ++j) {
                if (labels[j].compare(categories[i]) == 0) {
                    encoded_labels[i] = j;
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                cout << "ERROR: " << i << " " << categories[i] << endl;
            }
        }
    }
    return encoded_labels;
}

split_data split_train_test(std::vector<std::string> audio_files, std::vector<int> enc_categories, int percent_test) {
    // get the numbers and shuffle them
    int num_files = audio_files.size();
    int* indices = (int*)malloc(num_files*sizeof(int));
    for (int i=0; i<num_files; ++i) {
        indices[i] = i;
    }
    shuffle(indices, num_files, 10);

    int num_test = (int)num_files*(percent_test/100.0f);
    int num_training = num_files - num_test;
    printf("%d Test\n%d Training\n", num_test, num_training);

    split_data sd = split_data();
    for (int i=0; i<num_training; ++i) {
        sd.training_filenames.push_back(audio_files[indices[i]]);
        sd.training_categories.push_back(enc_categories[indices[i]]);
    }
    for (int i=0; i<num_test; ++i) {
        sd.test_filenames.push_back(audio_files[indices[i+num_training]]);
        sd.test_categories.push_back(enc_categories[indices[i+num_training]]);
    }

    free(indices);

    return sd;
}

int main() {
    auto start_t = std::chrono::high_resolution_clock::now();

    std::ifstream file("../c_implementation/tsetlin/env_data/ESC-50-master/meta/esc50.csv"); // Open the file
    if (!file) {
        std::cerr << "Error: File could not be opened!" << std::endl;
        return 1;
    }

    std::vector<std::string> filenames;
    std::vector<std::string> classes;

    std::string line;
    while (std::getline(file, line)) { // Read line by line
        int loc = 0;
        std::stringstream ss(line);
        std::string token;
        std::string filename;
        std::string category;
        char delim = ',';
        while(std::getline(ss, token, delim)) {
            if (loc == 0) {
                filename = token;
            }

            if (loc == 3) {
                category = token;
            }

            if (loc == 4 && strcmp("True", token.c_str()) == 0) {
                filenames.push_back(filename);
                classes.push_back(category);
            }
            loc += 1;
        }
    }
    file.close();

    auto test_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_d = test_t - start_t;
    cout << "Milli: " << ms_d.count() << endl;
    double secs1 = ms_d.count()/1000;
    cout << "Seconds: " << secs1 << endl;

    // encode classes
    int num_classes = 10;
    auto enc_classes = encode_labels(classes, num_classes);

    // Need to split up the files for test and train
    split_data sd = split_train_test(filenames, enc_classes, 20);

    int num_training = sd.training_filenames.size();
    int num_test = sd.test_filenames.size();
    std::vector<std::vector<float>> train_features(sd.training_filenames.size(), std::vector<float>());
    for (int i=0; i<num_training; ++i) {
        train_features[i] = read_file_and_preprocess(sd.training_filenames[i]);
    }
    
    // Binarize
    int bin_resolution = 25; 
    auto bin_thresholds = get_binarization_thresholds(train_features, bin_resolution);
    auto bin_train_features = binarize_features(train_features, bin_thresholds, bin_resolution);

    // output the training stuff
    std::ofstream outfile("cpp_features_training.txt");

    // Could eventually do a binary file
    if(outfile.is_open()) {
        outfile << num_training << endl;
        outfile << bin_train_features[0].size() << endl;
        outfile << num_classes << endl;
        outfile << std::fixed << std::setprecision(2);
        for (int i=0; i<num_training; ++i) {
            for (int j=0; j<bin_train_features[0].size(); ++j) {
                if (bin_train_features[i][j] == 1) {
                    outfile << 1 << " ";
                } else {
                    outfile << 0 << " ";
                }
            }
            outfile << sd.training_categories[i] << endl;
        }
        outfile.close();

    } else {
        std::cerr << "Error: unable to open file" << endl;
    }

    auto endtrain_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = endtrain_t - start_t;
    cout << "Milli: " << ms_double.count() << endl;
    double secs = ms_double.count()/1000;
    cout << "Seconds: " << secs << endl;

    // then preprocess and binarize and output TEST stuff so I can test with my model
    std::vector<std::vector<float>> test_features(sd.test_filenames.size(), std::vector<float>());
    for (int i=0; i<num_test; ++i) {
        test_features[i] = read_file_and_preprocess(sd.test_filenames[i]);
        cout << test_features[i][0] << " done with " << i << endl;
    } 
    auto bin_test_features = binarize_features(test_features, bin_thresholds, bin_resolution);

    auto endpptest_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pptd = endpptest_t - endtrain_t;
    cout << "Milli: " << pptd.count() << endl;
    double secs2 = pptd.count()/1000;
    cout << "Seconds total testing, divide by 80: " << secs2 << endl;

    // output the test stuff
    std::ofstream outfiletest("cpp_features_test.txt");

    // Could eventually do a binary file
    if(outfiletest.is_open()) {
        outfiletest << num_test << endl;
        outfiletest << bin_test_features[0].size() << endl;
        outfiletest << num_classes << endl;
        for (int i=0; i<num_test; ++i) {
            for (int j=0; j<bin_test_features[0].size(); ++j) {
                if (bin_test_features[i][j] == 1) {
                    outfiletest << 1 << " ";
                } else {
                    outfiletest << 0 << " ";
                }
            }
            outfiletest << sd.test_categories[i] << endl;
        }
        outfiletest.close();

        cout << "done writing" << endl;
        cout << "Num bin test features: " << bin_test_features[0].size() << endl;
    } else {
        std::cerr << "Error: unable to open file" << endl;
    }
}

