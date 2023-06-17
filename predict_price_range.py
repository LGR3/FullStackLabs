# IMPORTs
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import joblib

# Load .joblib Model
model = joblib.load('trained_model.joblib')

categorical_cols = ['homeType', 'hasSpa', 'city']

# PreProcessing function
def preprocess_data(data):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    categories = [data[col].unique() for col in categorical_cols]
    encoded_features = [f"{col}_{cat}" for i, col in enumerate(categorical_cols) for cat in categories[i]]
    X_encoded.columns = encoded_features
    X = pd.concat([X_encoded, data.drop(categorical_cols + ['priceRange'], axis=1)], axis=1)
    
    return encoder, encoded_features, X


# PREDICT Function
def predict_price_range(new_data, encoder, encoded_features):
    new_data_encoded = pd.DataFrame(encoder.transform(new_data[categorical_cols]))
    new_data_encoded.columns = encoded_features
    new_data_preprocessed = pd.concat([new_data_encoded, new_data.drop(categorical_cols + ['description', 'priceRange', 'uid'], axis=1)], axis=1)
    
    predictions = model.predict(new_data_preprocessed)
    
    return predictions


# Main function
if __name__ == '__main__':
    # load data
    data = pd.read_csv('train.csv')
    
    # preprocess
    encoder, encoded_features, X = preprocess_data(data)
    
    # test function
    predictions = predict_price_range(data, encoder, encoded_features)
    
    print(predictions)



# EXTRA
# Main function with sys.args for DEPLOY in CLI Cloud systems

# if __name__ == '__main__':
#     data_file = sys.argv[1]
#     predict_big5()