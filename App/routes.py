from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
import numpy as np
from scripts.data_collection import save_user_behavior, collect_video_data
from scripts.data_preprocessing import remove_duplicates
from scripts.predict import preprocess_input_data, recommend_top_videos
from tensorflow.keras.models import load_model
import json
from scripts.untils import parse_duration, read_video_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Tải mô hình đã huấn luyện và biên dịch lại với các metrics
model = load_model('cnn_content_filtering_model.keras')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def search_videos(query):
    youtube = build('youtube', 'v3', developerKey='AIzaSyDzB3OTNgkr5eGKb9qbvG2N3VWFZMcuLF8', cache_discovery=False)
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=10
    ).execute()
    
    search_results = search_response.get('items', [])
    
    # Lưu kết quả tìm kiếm vào file video_data.txt
    with open('Data/video_data.txt', 'a', encoding='utf-8') as f:
        for video in search_results:
            f.write(json.dumps(video) + '\n')
    
    return search_results

@app.route('/')
def index():
    # Call the update_recommendations route
    response = update_recommendations()
    if response.status_code != 200:
        return response

    # Read recommended videos from file
    try:
        with open('Data/recommended_videos.txt', 'r', encoding='utf-8') as f:
            recommended_videos = [json.loads(line) for line in f]
    except Exception as e:
        return f"Error reading recommended videos: {e}", 500

    return render_template('index.html', recommendations=recommended_videos)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return "Query is required", 400
    
    # Gọi hàm tìm kiếm video
    search_results = search_videos(query)
    
    return render_template('index.html', search_results=search_results)

@app.route('/watch/<video_id>')
def watch(video_id):
    # Gọi route cập nhật dữ liệu đề xuất
    response = update_recommendations()
    if response.status_code != 200:
        return response

    # Đọc dữ liệu video từ file video_data.txt
    video_data = read_video_data('Data/video_data.txt')
    
    # Tìm video theo ID
    video = next((video for video in video_data if video.get('id', {}).get('videoId') == video_id), None)
    if not video:
        return "Video not found", 404
    
    return render_template('watch.html', video=video)

@app.route('/log_watch_time', methods=['POST'])
def log_watch_time():
    data = request.get_json()
    video_id = data.get('video_id')
    watch_time = data.get('watch_time')
    
    if not video_id or watch_time is None:
        return jsonify({"error": "Video ID and Watch Time are required"}), 400
    
    # Lưu dữ liệu người dùng
    save_user_behavior('user_id_placeholder', video_id, watch_time, 'watch')
    
    return jsonify({"message": "Watch time logged successfully"}), 200

@app.route('/update', methods=['POST'])
def update_recommendations():
    video_data = read_video_data('Data/video_data.txt')
    video_data = remove_duplicates(video_data)
    
    # Tiền xử lý dữ liệu và dự đoán
    text_data = [video['snippet']['title'] + ' ' + video['snippet']['description'] for video in video_data]
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    X_text = vectorizer.transform(text_data).toarray()
    
    # Pad sequences to ensure uniform length
    max_length = 100
    X_text = pad_sequences(X_text, maxlen=max_length, padding='post')
    
    views = [int(video['statistics']['viewCount']) if 'statistics' in video else 0 for video in video_data]
    likes = [int(video['statistics']['likeCount']) if 'statistics' in video else 0 for video in video_data]
    durations = [parse_duration(video['contentDetails']['duration']) if 'contentDetails' in video else 0 for video in video_data]
    
    numerical_features = np.column_stack((views, likes, durations))
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_numerical = scaler.transform(numerical_features)
    
    predictions = model.predict([X_text, X_numerical])
    user_id = 'user_id_placeholder'  # Thay thế bằng user_id thực tế nếu có
    top_videos = recommend_top_videos(user_id, model, video_data, predictions)
    
    return jsonify(top_videos)

if __name__ == '__main__':
    app.run(debug=True)