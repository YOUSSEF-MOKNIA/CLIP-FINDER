from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from video_processing import video_scene_search, sanitize_filename
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello, World!"



@app.route('/process_videos', methods=['POST'])
def process_videos():
    links = request.json['links']
    description = request.json['description']
    results = video_scene_search(links, description)

    # Add sanitized_video_name to each result
    for result in results:
        result['sanitized_video_name'] = sanitize_filename(result['video_name']) 

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 