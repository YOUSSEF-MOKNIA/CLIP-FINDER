# ClipFinder : Video Scene Searching Based on Frame Watermarking

## Overview

Ever spent ages trying to find a specific moment in a video? **ClipFinder** helps you do that!
This project focuses on enabling efficient scene searching within a single video. By applying frame watermarking and object detection techniques, we aim to provide a way to quickly locate specific moments without the need to watch the entire video.

## How It Works

- **Frame Extraction and Analysis:** Key frames representing distinct scenes are extracted from the video based on significant changes between frames.
- **Frame Watermarking:** Unique identifiers are subtly embedded into these key frames using Discrete Cosine Transform (DCT), ensuring the information is hidden within the image data.
- **Object Detection:** A pre-trained object detection model (YOLOv8) is utilized to identify and classify objects present in each key frame.
- **Database Creation:** All this scene info and the frame markers are stored in a MongoDB database.
- **Search Functionality:** Users can then search for scenes using keywords or descriptions.
- **Result Presentation:** Finally, we show the user a list of the matching scenes so they can easily pick what they need.

## How to Use

Instructions will be provided upon full implementation of the project.

## Dependencies

- OpenCV
- NumPy
- Pandas
- PyMongo (for MongoDB)
- PyTorch/torchvision (if using YOLOv8)
- tqdm (for progress bars)
- NLTK (for natural language processing)
- scikit-learn (for TF-IDF and cosine similarity)
    
## Acknowledgements

- This project **ClipFinder** was developed as part of "Video Scene Searching Based on Frame Watermarking" at FP OUARZAZATE BY **MOKNIA Youssef**.
