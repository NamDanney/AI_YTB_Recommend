import json
import csv

def read_video_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        video_data = [json.loads(line) for line in f]
    return video_data

def read_user_data(file_path):
    user_data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                user_id, video_id, watch_time, action = row
                if user_id not in user_data:
                    user_data[user_id] = []
                user_data[user_id].append({
                    'video_id': video_id,
                    'watch_time': int(watch_time),
                    'action': action
                })
    except Exception as e:
        print(f"Error reading user data: {e}")
    return user_data

def parse_duration(duration):
    # Handle different duration formats
    if duration.startswith('PT'):
        duration = duration[2:]
    elif duration.startswith('P'):
        duration = duration[1:]
    else:
        return 0

    hours = 0
    minutes = 0
    seconds = 0

    if 'H' in duration:
        hours, duration = duration.split('H')
        hours = int(hours) if hours else 0
    if 'M' in duration:
        minutes, duration = duration.split('M')
        minutes = int(minutes) if minutes else 0
    if 'S' in duration:
        seconds = int(duration.replace('S', '')) if duration else 0

    total_seconds = hours * 3600 + minutes * 60 + seconds
    
    return total_seconds