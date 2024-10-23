import json
from flask import Flask
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from scripts.untils import parse_duration
from scripts.untils import read_video_data, read_user_data
import csv
import pickle

app = Flask(__name__)

def preprocess_input_data(video_data, vectorizer, scaler, max_length=100):
    text_data = [video['snippet']['title'] + ' ' + video['snippet']['description'] for video in video_data]
    X_text = vectorizer.transform(text_data).toarray()
    
    # Pad sequences to ensure uniform length
    X_text = pad_sequences(X_text, maxlen=max_length, padding='post')
    
    views = [int(video['statistics']['viewCount']) if 'statistics' in video else 0 for video in video_data]
    likes = [int(video['statistics']['likeCount']) if 'statistics' in video else 0 for video in video_data]
    durations = [parse_duration(video['contentDetails']['duration']) if 'contentDetails' in video else 0 for video in video_data]
    
    numerical_features = np.column_stack((views, likes, durations))
    X_numerical = scaler.transform(numerical_features)
    
    return X_text, X_numerical

def make_predictions(model, X_text, X_numerical):
    return model.predict([X_text, X_numerical])

def recommend_top_videos(user_id, model, video_data, predictions, top_n=12):
    top_indices = np.argsort(predictions.flatten())[-top_n:]

    recommended_videos = []
    for idx in top_indices:
        video = video_data[int(idx)]
        recommended_videos.append({
            'id': video['id'],
            'snippet': {
                'title': video['snippet']['title'],
                'thumbnails': video['snippet']['thumbnails']
            }
        })

    with open('Data/recommended_videos.txt', 'w', encoding='utf-8') as file:
        for video in recommended_videos:
            file.write(json.dumps(video, ensure_ascii=False) + '\n')

    return recommended_videos

def read_user_data(filepath):
    user_data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:
                    user_data.append({
                        "user_id": row[0],
                        "video_id": row[1],
                        "watch_time": int(row[2]),
                        "action": row[3]
                    })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return user_data

def main():
    with app.app_context():
        video_data = read_video_data('Data/video_data.txt')
        user_data = read_user_data('Data/user_data.txt')
        
        # Load the fitted vectorizer and scaler
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        max_length = 100
        X_text, X_numerical = preprocess_input_data(video_data, vectorizer, scaler, max_length)
        
        model = load_model('cnn_content_filtering_model.keras')
        
        predictions = make_predictions(model, X_text, X_numerical)
        user_id = 'user_id_placeholder'
        recommended_videos = recommend_top_videos(user_id, model, video_data, predictions, top_n=12)    
        
        with open('Data/recommended_videos.txt', 'w', encoding='utf-8') as f:
            for video in recommended_videos:
                f.write(json.dumps(video) + '\n')
        
        print("Recommendations updated successfully")

if __name__ == '__main__':
    main()