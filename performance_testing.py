from sklearn.metrics import accuracy_score
from train_tsetlin import (
    load_and_featurize_training_data, 
    create_and_train_tsetlin_machine, 
    featurize_test_data,
    binarize_training_data
)
import time
from memory_profiler import memory_usage
import os
import psutil
from tm_file import (
    print_tm_to_file
)

def do_initial_load_and_featurization():
    start_time = time.time()
    start_time_p = time.process_time()
    x_train_featurized, x_test, y_train, y_test, scaler = load_and_featurize_training_data()
    end_time = time.time()
    end_time_p = time.process_time()

    process = psutil.Process(os.getpid())
    ram_used = process.memory_info().rss / (1024 * 1024)  # in MB
    print("RAM USED: ")
    print(round(ram_used, 2))

    elapsed_time = end_time - start_time
    p_time = end_time_p - start_time_p

    return x_train_featurized, x_test, y_train, y_test, scaler, elapsed_time, p_time

def time_training_and_prediction(
        binarization_resolution, clauses, T, s, epochs,
        x_train, x_test, y_train, y_test, scaler,
        load_and_feat_time, load_and_feat_time_p):
    # Time training
    start_time_train = time.time()
    start_time_train_p = time.process_time()
    x_train_feat_bin, thresholds = binarize_training_data(binarization_resolution, x_train)
    end_time_train_feat = time.time()
    end_time_train_feat_p = time.process_time()

    tm = create_and_train_tsetlin_machine(clauses, T, s, epochs, x_train_feat_bin, y_train)
    print("training done")
    #for testing
    process = psutil.Process(os.getpid())
    ram_used = process.memory_info().rss / (1024 * 1024)  # in MB
    print("RAM USED: ")
    print(round(ram_used, 2))

    end_time_train_tm = time.time()
    end_time_train_tm_p = time.process_time()

    # Time prediction on 1 piece of test data
    start_time_test1 = time.time()
    start_time_test1_p = time.process_time()
    x_test_feat1 = featurize_test_data([x_test[0]], scaler, thresholds)
    end_time_test1_feat = time.time()
    end_time_test1_feat_p = time.process_time()
    y_pred1 = tm.predict(x_test_feat1)
    end_time_test1_pred = time.time()
    end_time_test1_pred_p = time.process_time()

    # Time prediction for all test data
    start_time_testall = time.time()
    start_time_testall_p = time.process_time()
    x_test_feat_all = featurize_test_data(x_test, scaler, thresholds)
    end_time_testall_feat = time.time()
    end_time_testall_feat_p = time.process_time()
    y_predall = tm.predict(x_test_feat_all)
    end_time_testall_pred = time.time()
    end_time_testall_pred = time.process_time()

    training_time = (end_time_train_tm - start_time_train) + load_and_feat_time
    training_time_p = (end_time_train_tm_p - start_time_train_p) + load_and_feat_time_p

    train_feat_bin = (end_time_train_feat - start_time_train) + load_and_feat_time
    train_feat_bin_p = (end_time_train_feat_p - start_time_train_p) + load_and_feat_time_p
    train_tm = (end_time_train_tm - end_time_train_feat)
    train_tm_p = (end_time_train_tm_p - end_time_train_feat_p)

    testing_time = end_time_test1_pred - start_time_test1
    testing_time_p = end_time_test1_pred_p - start_time_test1_p

    print("Total time to train: ", training_time)
    print("Total time to train (process): ", training_time_p)
    print("Time to featurize/binarize: ", train_feat_bin)
    print("Time to featurize/binarize (process): ", train_feat_bin_p)
    print("Time to train tm: ", train_tm)
    print("Time to train tm (process): ", train_tm_p)
    print("\n")
    print("Total time to test 1: ", testing_time)
    print("Total time to test 1 (process): ", testing_time_p)
    print("Test time feat 1: ", end_time_test1_feat - start_time_test1)
    print("Test time feat 1 (process): ", end_time_test1_feat_p - start_time_test1_p)
    print("Test time pred 1: ", end_time_test1_pred - end_time_test1_feat)
    print("Test time pred 1 (process): ", end_time_test1_pred_p - end_time_test1_feat_p)

    accuracy = accuracy_score(y_test, y_predall)
    print(f"Clauses: {clauses}, T: {T}, s: {s}, epochs: {epochs}, binarization resolution: {binarization_resolution} -> Accuracy: {accuracy * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy, training_time, testing_time, training_time_p, testing_time_p

def run_specific_tests_and_output_csv():
    x_train, x_test, y_train, y_test, scaler, load_and_feat_time, load_and_feat_time_p = do_initial_load_and_featurization()
    print("Load and featurize data time: ", load_and_feat_time)
    print("Load and featurize data time (process): ", load_and_feat_time_p)
    # Clauses, T, s, epochs, bin res
    test_vals = [
        (400,50,4,200,25)
        ]

    for clause, T_val, s, epoch, bin_res in test_vals:
        time_training_and_prediction(bin_res, clause, T_val, s, epoch,
                                       x_train, x_test, y_train, y_test, scaler,
                                       load_and_feat_time, load_and_feat_time_p)
 
process = psutil.Process(os.getpid())
ram_used = process.memory_info().rss / (1024 * 1024)  # in MB
print("RAM USED: ")
print(round(ram_used, 2))

run_specific_tests_and_output_csv()
