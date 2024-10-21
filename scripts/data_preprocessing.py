import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scripts.untils import read_video_data, parse_duration
from .model import create_model

def preprocess_text_data(text_data, max_length=100):
    # Điều chỉnh các tham số max_df và min_df
    vectorizer = CountVectorizer(max_df=max_length, min_df=2, max_features=max_length)
    X_text = vectorizer.fit_transform(text_data)
    X_text = X_text.toarray()
    
    return X_text, vectorizer

def remove_duplicates(video_data):
    seen = set()
    unique_videos = []
    for video in video_data:
        video_id = video.get('id', {}).get('videoId')
        if video_id and video_id not in seen:
            unique_videos.append(video)
            seen.add(video_id)
    return unique_videos

def preprocess_data():
    video_data = read_video_data('Data/video_data.txt')
    video_data = remove_duplicates(video_data)
    
    text_data = [
        f"{video['snippet']['title']} {' '.join(video['snippet'].get('tags', []))}"
        for video in video_data
    ]
    X_text, vectorizer = preprocess_text_data(text_data)
    
    svd = TruncatedSVD(n_components=100, random_state=42)  # Ensure n_components is large enough
    X_text = svd.fit_transform(X_text)
    
    views = [int(video['statistics']['viewCount']) if 'statistics' in video else 0 for video in video_data]
    likes = [int(video['statistics']['likeCount']) if 'statistics' in video else 0 for video in video_data]
    durations = [parse_duration(video['contentDetails']['duration']) if 'contentDetails' in video else 0 for video in video_data]
    
    numerical_features = np.column_stack((views, likes, durations))
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(numerical_features)
    
    X_train_text, X_test_text, X_train_num, X_test_num = train_test_split(X_text, X_numerical, test_size=0.2, random_state=42)
    
    return X_train_text, X_test_text, X_train_num, X_test_num, vectorizer, scaler

# Gọi hàm preprocess_data để tiền xử lý dữ liệu
X_train_text, X_test_text, X_train_num, X_test_num, vectorizer, scaler = preprocess_data()

if __name__ == "__main__":
    vocab_size = len(vectorizer.vocabulary_)
    embedding_dim = 50
    num_numerical_features = X_train_num.shape[1]

    model = create_model((None,), (num_numerical_features,))
    model.fit([X_train_text, X_train_num], np.ones(len(X_train_text)), epochs=10, batch_size=32, validation_split=0.2)
    model.evaluate([X_test_text, X_test_num], np.ones(len(X_test_text)))

    model.save('cnn_content_filtering_model.h5')