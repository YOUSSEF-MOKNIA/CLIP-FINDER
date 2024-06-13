import pymongo

### Stores Data in MongoDB

def store_data_mongodb(detection_results_df, database_name, collection_name):
    """
    Stores frame information, timestamps, and detected object names in a MongoDB collection.

    Args:
        frame_info_df (pd.DataFrame): DataFrame containing frame names and timestamps.
        detection_results_df (pd.DataFrame): DataFrame with detection results.
        database_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection within the database.
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")  # Connect to local MongoDB instance
    db = client[database_name]
    collection = db[collection_name]

    for index, row in detection_results_df.iterrows():
        identifier = row["Frame Identifier"]
        frame_name = row["Frame Name"]
        timestamp = row["Timestamp (s)"]
        video_name = row["Video Name"]

        # Filter detection results for the current frame
        frame_detections = detection_results_df[detection_results_df["Frame Name"] == frame_name]

        # Extract object names
        object_names = []
        for _, detection in frame_detections.iterrows():
            objects = detection["Objects"]
            for obj in objects:
                object_names.append(obj["object_class"])

        # Create document to insert
        document = {
            "Frame Identifier": identifier,
            "Frame Name": frame_name,
            "Timestamp (s)": timestamp,
            "Video Name": video_name,
            "Objects": object_names
        }
        # Check if document already exists based on "Frame Identifier"
        existing_document = collection.find_one({"Frame Identifier": identifier}) 
        if not existing_document:
            # Insert document into collection only if it doesn't exist
            collection.insert_one(document)  
        else:
            print(f"Document with Frame Identifier '{identifier}' already exists. Skipping insertion.")

    print(f"Data inserted (or updated) into MongoDB: {database_name}.{collection_name}")
