from extract_features import load_data_and_split, extract_features_from_files
from normalize_and_binarize import normalize_training_features, normalize_test_features, get_binarization_thresholds, binarize_features
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

# Load all data. Featurize the training data.
# Return the training and test data
def load_and_featurize_training_data():
    x_train, x_test, y_train, y_test = load_data_and_split()

    x_train_features = extract_features_from_files(x_train)
    # Normalizing and Binarizing features
    scaler, x_train_features_norm = normalize_training_features(x_train_features)
    return x_train_features_norm, x_test, y_train, y_test, scaler

def binarize_training_data(binarization_resolution, x_train_features):
    thresholds  = get_binarization_thresholds(binarization_resolution, x_train_features)
    x_train_features_bin = binarize_features(x_train_features, thresholds)
    return x_train_features_bin, thresholds 

def create_and_train_tsetlin_machine(clauses, T, s, epochs, x_train, y_train):
    tm = MultiClassTsetlinMachine(number_of_clauses=clauses, T=T, s=s)
    tm.fit(x_train, y_train, epochs=epochs)
    return tm

def featurize_test_data(x_test, scaler, thresholds):
    x_test_features = extract_features_from_files(x_test)
    x_test_norm = normalize_test_features(scaler, x_test_features)
    x_test_bin = binarize_features(x_test_norm, thresholds)
    return x_test_bin
