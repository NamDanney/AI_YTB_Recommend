from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Concatenate, Input, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json
from scripts.data_preprocessing import remove_duplicates
from scripts.untils import parse_duration, read_video_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

def preprocess_text_data(text_data, max_length=100):
    # Tokenization and Vectorization
    vectorizer = CountVectorizer(max_features=max_length, stop_words='english')
    X_text = vectorizer.fit_transform(text_data).toarray()
    
    # Padding
    X_text = pad_sequences(X_text, maxlen=max_length, padding='post')
    
    return X_text, vectorizer

def preprocess_data():
    video_data = read_video_data('Data/video_data.txt')
    video_data = remove_duplicates(video_data)
    
    # Tiền xử lý dữ liệu văn bản và số liệu
    text_data = [video['snippet']['title'] + ' ' + video['snippet']['description'] for video in video_data]
    vectorizer = CountVectorizer(max_df=0.8, min_df=2)
    X_text = vectorizer.fit_transform(text_data).toarray()
    
    views = [int(video['statistics']['viewCount']) for video in video_data]
    likes = [int(video['statistics']['likeCount']) for video in video_data]
    durations = [parse_duration(video['contentDetails']['duration']) if 'contentDetails' in video else 0 for video in video_data]
    
    numerical_features = np.column_stack((views, likes, durations))
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(numerical_features)
    
    X_train_text, X_test_text, X_train_num, X_test_num = train_test_split(X_text, X_numerical, test_size=0.2, random_state=42)
    
    # Save the fitted vectorizer and scaler
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_text, X_test_text, X_train_num, X_test_num, vectorizer, scaler, video_data

def build_model(input_shape_text, input_shape_num):
    # Text input and CNN layers
    text_input = Input(shape=input_shape_text, name='text_input')
    embedding = Embedding(input_dim=10000, output_dim=128)(text_input)
    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding)
    pool1 = GlobalMaxPooling1D()(conv1)
    dense1 = Dense(128, activation='relu')(pool1)

    # Numerical input and Dense layers
    num_input = Input(shape=input_shape_num, name='num_input')
    dense2 = Dense(64, activation='relu')(num_input)
    dense3 = Dense(32, activation='relu')(dense2)

    # Combine CNN and numerical features
    combined = Concatenate()([dense1, dense3])
    final_output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[text_input, num_input], outputs=final_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    X_train_text, X_test_text, X_train_num, X_test_num, vectorizer, scaler, video_data = preprocess_data()
    
    vocab_size = len(vectorizer.vocabulary_)
    embedding_dim = 50
    num_numerical_features = X_train_num.shape[1]
    max_length = 100  # Define max_length here
    model = build_model((max_length,), (num_numerical_features,))
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    
    # Train the model with validation data and callbacks
    history = model.fit(
        [X_train_text, X_train_num], 
        np.ones(len(X_train_text)), 
        epochs=10, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stopping, model_checkpoint]
    )
    
    model.evaluate([X_test_text, X_test_num], np.ones(len(X_test_text)))

    model.save('cnn_content_filtering_model.keras')
    
    return model, X_test_text, X_test_num, history

def evaluate_model(model, X_test_text, X_test_num, y_test):
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict([X_test_text, X_test_num])
    y_pred = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return accuracy, precision, recall, f1

def check_accuracy(accuracy):
    if (accuracy >= 0.70):
        print("Model accuracy meets the requirement.")
    else:
        print("Model accuracy does not meet the requirement.")

if __name__ == "__main__":
    try:
        model, X_test_text, X_test_num, history = train_model()
        y_test = np.ones(len(X_test_text))  # Dummy labels for illustration
        accuracy, precision, recall, f1 = evaluate_model(model, X_test_text, X_test_num, y_test)
        check_accuracy(accuracy)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()