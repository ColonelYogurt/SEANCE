import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import welch
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout

# Load EEG data


def load_eeg_data(file_path):
    return pd.read_csv(file_path)

# Preprocess EEG data


def preprocess_eeg_data(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

# Extract features using PSD

# Used Welchs method to calc nperseg


def extract_features(eeg_data, fs=250, nperseg=512):
    psd_features = []
    for i in range(eeg_data.shape[1]):
        freqs, psd = welch(eeg_data[:, i], fs, nperseg=nperseg)
        psd_features.append(psd)
    return np.column_stack(psd_features)

# Validate labels


def validate_labels(labels, expected_labels):
    unique_labels = set(labels)
    if not unique_labels.issubset(set(expected_labels)):
        raise ValueError("Unexpected labels found")
    return labels

# CNN model


def build_enhanced_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3,
              activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Main
if __name__ == "__main__":
    expected_labels = ['yes', 'no', 'left',
                       'right', 'up', 'down', 'w', 'a', 's', 'd']
    file_path = 'path_to_your_eeg_data.csv'
    eeg_data = load_eeg_data(file_path)
    features, labels = preprocess_eeg_data(eeg_data)
    labels = validate_labels(labels, expected_labels)

    features_extracted = extract_features(features)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_extracted, encoded_labels, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build and train the model
    num_classes = len(np.unique(encoded_labels))
    input_shape = (features_extracted.shape[1], 1)
    cnn_model = build_enhanced_cnn_model(input_shape, num_classes)
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32,
                  validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100}%")
