import json
from googleapiclient.discovery import build
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, filename='error.log')


# Replace with your actual API key
api_key = 'AIzaSyDzB3OTNgkr5eGKb9qbvG2N3VWFZMcuLF8'
youtube = build('youtube', 'v3', developerKey=api_key)

def read_video_data(file_path):
    video_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                video = json.loads(line)
                logging.info(f"Read video data: {video}")
                video_data.append(video)
    return video_data

def get_video_details(video_id):
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        logging.info(f"API response for video ID {video_id}: {response}")
        if response['items']:
            video = response['items'][0]
            video_data = {
                'id': video['id'],
                'title': video['snippet']['title'],
                # 'tags': video['snippet'].get('tags', []),
                'viewCount': video['statistics'].get('viewCount', 0),
                'likeCount': video['statistics'].get('likeCount', 0),
                'duration': video['contentDetails']['duration'],
                'snippet': video['snippet'],  # Ensure snippet is included
                'statistics': video['statistics']  # Ensure statistics is included
            }
            return video_data
        else:
            logging.warning(f"No items found for video ID: {video_id}")
            return None
    except Exception as e:
        logging.error(f"An error occurred while retrieving video details: {e}")
        return None

def collect_video_data(video_ids):
    video_data = []
    for video_id in video_ids:
        video_details = get_video_details(video_id)
        if video_details:
            video_data.append(video_details)
    return video_data

def save_video_data(video_data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:  # Open file in append mode
        for video in video_data:
            logging.info(f"Saving video data: {video}")
            file.write(json.dumps(video, ensure_ascii=False) + '\n')


def write_video_data(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:  
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_user_behavior(user_id, video_id, watch_time, interaction):
    with open('Data/user_data.txt', 'a', encoding='utf-8') as file:
        file.write(f"{user_id},{video_id},{watch_time},{interaction}\n")