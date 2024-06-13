# Import Libraries:
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2, pafy
from pytube import YouTube
import re
from search import process_search_query, search_by_keywords
from database_mongo import store_data_mongodb

# Extract the Youtube video id from the link 

def extract_video_id(youtube_link):
    index = youtube_link.find('=')

    if index != -1 :
        youtube_link = youtube_link[index+1:]

    return youtube_link

### Video name Extraction
    
def get_video_title_or_filename(video_source):
    """Extracts video title from URL or filename from local path."""
    if video_source.startswith(("http", "www")):
        try:
            video = pafy.new(video_source)
            return video.title
        except Exception as e:
            print(f"Error getting video title: {e}")
            return None

    else:
        # Local file: Extract filename without extension
        return os.path.splitext(os.path.basename(video_source))[0]


### Remove or Replace invalide characters in the Video Name

def sanitize_filename(filename):
    invalid_chars = r'[\\\/:*?"<>|\' \( \)]'  # Characters not allowed in Windows filenames
    sanitized_filename = re.sub(invalid_chars, '', filename)  # Replace with underscores
    return sanitized_filename

## Video Loading
def video_load_from_url(url, download = False) :
    
    if download == True :
        try:
            streams = YouTube(url).streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)
            video_name = get_video_title_or_filename(url)
            sanitized_video_name = sanitize_filename(video_name)
            video_name_ex = f"{sanitized_video_name},.mp4"
            video_path = os.path.join("Videos", video_name_ex)
            streams[0].download(filename=video_path)
            url = extract_video_id(url)
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None, None

    elif download == False :
        try:
            url = extract_video_id(url)
            video = pafy.new(url)
            best = video.getbest(preftype = 'mp4')
            cap = cv2.VideoCapture(best.url)
        except Exception as e:
            print(f"Error accessing video: {e}")
            return None, None
    
    return cap, url

def get_video_input(user_input):
    if user_input.startswith(("http", "www")):
        return video_load_from_url(user_input, download=True)
    else:
        # Treat as local file path
        if os.path.isfile(user_input):
            cap = cv2.VideoCapture(user_input)
            return cap, user_input
        else:
            print("Invalid file path. Please try again.")


## Frame Extraction
def extract_frames_with_changes(input_source, sanitized_video_name, output_folder, threshold=0.1):
    """Extracts frames with significant changes from a video, saves them with timestamps, and displays a progress bar.

    Args:
        video_path (str): The path to the video file.
        output_folder (str): The folder to save extracted frames.
        threshold (float, optional): Threshold for frame difference detection (0.0-1.0). Defaults to 0.1.

    Returns:
        pd.DataFrame: DataFrame containing frame names and timestamps.
    """
    # video load
    print('video load ... ')
    # Example usage:
    cap, video_source = get_video_input(input_source)
    print('video load complete ')
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    frame_count = 0  # Initialize frame counter
    previous_frame = None  # Variable to store previous frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    frame_info = []  # List to store frame names and timestamps
    video_name = get_video_title_or_filename(input_source)
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:  # Initialize tqdm progress bar
        while True:
            ret, frame = cap.read()  # Read frame from video
            if not ret:  # Break loop if no more frames are available
                break

            if previous_frame is not None:  # Check if previous frame exists
                difference = cv2.absdiff(frame, previous_frame).astype("uint8")  # Calculate absolute difference between frames
                difference_mean = np.mean(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY))  # Calculate mean difference
                if difference_mean > threshold * 255:  # Check if difference exceeds threshold
                    frame_name = f"{sanitized_video_name}_frame_{frame_count:04d}.jpg"  # Generate frame filename
                    cv2.imwrite(os.path.join(output_folder, frame_name), frame)  # Save frame to output folder
                    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get frame timestamp (in seconds)
                    frame_info.append((frame_name, frame_time, video_name))  # Add frame info to list
                    frame_count += 1  # Increment frame counter

            previous_frame = frame  # Update previous frame
            pbar.update(1)  # Update progress bar

    cap.release()  # Release video capture object
    print(f"\nExtracted {frame_count} frames with changes to {output_folder}")

    # Create DataFrame from frame information
    df = pd.DataFrame(frame_info, columns=["Frame Name", "Timestamp (s)", "Video Name"])
    return df


## Information Embedding
def embed_identifiers(frames_folder, output_folder):
    """Embeds unique numbers as identifiers into the DCT coefficients of each frame."""
    os.makedirs(output_folder, exist_ok=True)
    frame_info_df = pd.DataFrame()

    for i, frame_file in enumerate(tqdm(os.listdir(frames_folder), desc="Embedding Identifiers")):
        if frame_file.endswith(".jpg"):  # Assuming frames are in JPEG format
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Generate unique identifier
            identifier = str(i)

            # Embed identifier into DCT coefficients
            embed_identifier(frame, identifier)

            # Save the frame with embedded identifier
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, frame)

            # Append frame info to DataFrame
            frame_info_df = frame_info_df.append({"Frame Name": frame_file, "Identifier": identifier}, ignore_index=True)

    print(f"Embedded identifiers in frames saved to {output_folder}")

def embed_identifier(frame, identifier):
    """Embeds an identifier into the DCT coefficients of a frame."""
    # Prepare identifier for embedding
    data_to_embed = identifier  # Use the identifier directly
    data_binary = ''.join(format(ord(char), '08b') for char in data_to_embed)

    # Embed data into DCT coefficients
    data_idx = 0
    for i in range(3):  # Process each color channel
        channel = frame[:, :, i].astype(np.float32) - 128
        dct = cv2.dct(channel)
        for k in range(len(data_binary)):
            if data_idx < len(dct.flatten()):
                dct.flatten()[k] += int(data_binary[k])  # Modify DCT coefficients
                data_idx += 1
            else:
                break


### Embedded Identifier Extraction

def extract_identifiers(embedded_frames_folder):
    """Extracts identifiers from the DCT coefficients of each embedded frame."""
    frame_info_list = []

    for frame_file in tqdm(os.listdir(embedded_frames_folder), desc="Extracting Identifiers"):
        if frame_file.endswith(".jpg"):  # Assuming frames are in JPEG format
            frame_path = os.path.join(embedded_frames_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Extract identifier from DCT coefficients
            identifier = extract_identifier(frame)

            # Append frame info to the list
            frame_info_list.append({"Frame Name": frame_file, "Identifier": identifier})

    # Create a DataFrame from the list of extracted identifiers
    frame_info_df = pd.DataFrame(frame_info_list)
    return frame_info_df

def extract_identifier(frame):
    """Extracts the identifier from the DCT coefficients of a frame."""
    identifier_bits = ''
    data_idx = 0
    for i in range(3):  # Process each color channel
        channel = frame[:, :, i].astype(np.float32) - 128
        dct = cv2.dct(channel)
        for val in dct.flatten():
            if data_idx < 32:  # We embed 32 bits for the identifier
                identifier_bits += str(int(val) & 1)  # Extract the LSB of each coefficient
                data_idx += 1
            else:
                break

    identifier = int(identifier_bits, 2)
    return identifier


## Object Detecting
def detect_objects_in_embedded_frames(frames_folder, frame_info_df, model_path="yolov8l.pt"):
    """
    Detects objects in embedded frames and returns a DataFrame with frame name, timestamp, and detected objects.

    Args:
        frames_folder (str): The folder containing the embedded frames.
        frame_info_df (pd.DataFrame): DataFrame containing frame names and timestamps.
        model_path (str, optional): Path to the trained YOLOv8 model. Defaults to "yolov8l.pt".

    Returns:
        pd.DataFrame: DataFrame containing frame names, timestamps, and detected object information.
    """

    # Initialize YOLOv8 model
    model = YOLO(model_path)

    # Initialize empty list to store results
    results_list = []

    # Iterate through each frame in the folder
    for index, frame_name in enumerate(tqdm(os.listdir(frames_folder), desc="Detecting Objects")):
        # Check if it's an image file
        if not frame_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # Open the frame
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        # Detect objects in the frame
        results = model.predict(frame)

        # Get the correct timestamp for this frame using the index
        timestamp = frame_info_df['Timestamp (s)'][index]  # Access specific timestamp
        frame_identifier = frame_info_df['Identifier'][index]
        video_name = frame_info_df['Video Name'][index]    
        # Store detected objects for the current frame
        frame_objects = []
        objects_coords = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                class_id = box.cls[0].item()
                prob = round(box.conf[0].item(), 2)
                frame_objects.append({
                    "object_class": result.names[box.cls[0].item()]
                })
                objects_coords.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": prob
                })

        # Append information for the frame with all detected objects
        results_list.append({
            "Frame Identifier": frame_identifier,
            "Video Name": video_name,
            "Frame Name": frame_name,
            "Timestamp (s)": timestamp,  # Use the extracted timestamp
            "Objects": frame_objects, # Store a list of objects for each frame
            "Coords": objects_coords
        })

    # Create and return DataFrame from the results list
    df = pd.DataFrame(results_list)
    return df


## Saves Visualizes object detections on frames
def visualize_detections(frame_info_df, results_df, frames_folder, output_folder):
    """
    Visualizes object detections on frames and saves them to a new folder.

    Args:
        frame_info_df (pd.DataFrame): DataFrame containing frame names and timestamps.
        results_df (pd.DataFrame): DataFrame containing detection results.
        frames_folder (str): The folder containing the original frames.
        output_folder (str): The folder to save the frames with visualizations.
    """

    os.makedirs(output_folder, exist_ok=True)  # Create output folder

    for index, row in tqdm(frame_info_df.iterrows(), desc="Visualizing Detections", total=len(frame_info_df)):
        frame_name = row["Frame Name"]
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        # Filter detection results for the current frame
        frame_detections = results_df[results_df["Frame Name"] == frame_name]

        for _, detection in frame_detections.iterrows():
            coords = detection["Coords"]
            objects = detection["Objects"]

            for i, obj in enumerate(objects):  # Use enumerate with 'objects' list
                coord = coords[i]  # Access corresponding coordinate from 'coords'
                x1 = coord.get("x1", 0)  # Get coordinates, default to 0 if not present
                y1 = coord.get("y1", 0)
                x2 = coord.get("x2", 0)
                y2 = coord.get("y2", 0)
                class_name = obj["object_class"]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the visualized frame
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, frame)

    print(f"Frames with detections saved to {output_folder}")


### Plot the frame with bounding boxes and labels of the detected objects.

def plot_image(image_path):
    """
    Plots an image using matplotlib.

    Args:
        image_path (str): The path to the image file.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis ticks and labels
    plt.title(f"Image: {image_path}")
    plt.show()


## Test the search with Keywords
def video_scene_search(video_urls, user_query):
    """
    Performs scene search on a list of videos based on a user query.

    Args:
        video_urls (list): A list of YouTube video URLs or local video file paths.
        user_query (str): The user's search query (keywords or description).

    Returns:
        list: A list of dictionaries containing matching frame information 
              (video name, frame path, timestamp) for each video.
    """

    all_results = []

    for video_url in video_urls:
        try:
            # 1. Video Loading and Frame Extraction
            video_name = get_video_title_or_filename(video_url)
            sanitized_video_name = sanitize_filename(video_name)
            extracted_frames_folder = os.path.join("Frames/extracted_frames", sanitized_video_name)
            frame_info_df = extract_frames_with_changes(video_url, sanitized_video_name, extracted_frames_folder)

            # 2. Information Embedding and Object Detection
            embedded_frames_folder = os.path.join("Frames/embedded_frames", sanitized_video_name)
            embed_identifiers(extracted_frames_folder, embedded_frames_folder)
            extracted_frame_info_df = extract_identifiers(embedded_frames_folder)
            combined_df = pd.merge(extracted_frame_info_df, frame_info_df, on='Frame Name')
            detection_results_df = detect_objects_in_embedded_frames(embedded_frames_folder, combined_df)

            # 3. Visualization (Optional)
            visualized_frames_folder = os.path.join("Frames/visualized_frames", sanitized_video_name)
            visualize_detections(frame_info_df, detection_results_df, embedded_frames_folder, visualized_frames_folder)

            # 4. Store Data in MongoDB (Optional, for repeated searches)
            store_data_mongodb(detection_results_df, "video_search_db", "video_scenes") 

            # 5. Search
            keywords = process_search_query(user_query)
            matching_frames = search_by_keywords(keywords, extracted_frame_info_df, "video_search_db", "video_scenes")
            
            for frame in matching_frames:
                frame['video_name'] = video_name  # Add video name to results
            all_results.extend(matching_frames)

        except Exception as e:
            print(f"Error processing video {video_url}: {e}")
            
    
    return all_results
