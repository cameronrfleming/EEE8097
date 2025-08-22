import numpy as np
from sklearn.preprocessing import StandardScaler

# Normalize training features
def normalize_training_features(X_train_raw):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    return scaler, X_train_scaled

# Use input scalar from training to normalise test data
def normalize_test_features(scaler, X_test_raw):
    X_test_scaled = scaler.transform(X_test_raw)
    return X_test_scaled

# Use the resolution parameter and normalized training data
# to compute thresholds
def get_binarization_thresholds(resolution, x_train_scaled):
    # Binarization parameters
    num_features = x_train_scaled.shape[1]

    # Compute thresholds from training data
    feature_mins = x_train_scaled.min(axis=0)
    feature_maxs = x_train_scaled.max(axis=0)
    thresholds = [
        np.linspace(feature_mins[i], feature_maxs[i], num=resolution, endpoint=False)
        for i in range(num_features)
    ]

    return thresholds

# Use computed thresholds to binarize the features
def binarize_features(X, thresholds):
    num_features = X.shape[1]
    binarized_features = []
    for i in range(X.shape[0]):
        sample_features = []
        for j in range(num_features):
            binarized = X[i, j] > thresholds[j]
            sample_features.extend(binarized.astype(np.uint8))
        binarized_features.append(sample_features)

    return np.array(binarized_features)
