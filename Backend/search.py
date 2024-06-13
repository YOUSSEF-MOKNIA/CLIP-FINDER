## Search Functionality
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo


def process_search_query(query):
    """Processes a user query, extracting keywords from both comma-separated keywords and descriptions.

    Args:
        query (str): The user's search query.

    Returns:
        list: A list of extracted keywords.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    keywords = []

    # Handle comma-separated keywords
    if "," in query:
        keywords = [keyword.strip().lower() for keyword in query.split(",")]
    else:
        # Process description for keywords
        tokens = nltk.word_tokenize(query.lower())
        tokens = [token for token in tokens if token.isalnum()]  # Remove punctuation
        tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
        keywords = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize

    return keywords

#### Searches for matching frames based on keywords in the detected objects.
def search_by_keywords(keywords, frame_info_df, database_name, collection_name):
    """Searches for matching frames based on keywords and returns frame name, video name, and timestamp."""

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    collection = db[collection_name]

    # Build MongoDB query
    query = {"$and": []}
    for keyword in keywords:
        query["$and"].append({"Objects": keyword})  # Make sure "objects" field name matches your database

    # Retrieve matching frames from the database
    matching_frames = collection.find(query)

    results = []
    for frame_data in matching_frames:
        frame_identifier = frame_data["Frame Identifier"]
        matching_row = frame_info_df[frame_info_df["Identifier"] == frame_identifier]
        if not matching_row.empty:
            video_name = frame_data["Video Name"]  # Assuming collection name is video name
            frame_name = matching_row["Frame Name"].iloc[0]
            timestamp = frame_data["Timestamp (s)"]  # Assuming you have the timestamp in the database
            results.append({
                'video_name': video_name,
                'frame_path': frame_name,
                'timestamp': timestamp
            })
                                
    return results