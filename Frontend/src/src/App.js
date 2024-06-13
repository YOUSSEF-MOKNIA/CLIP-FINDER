import React, { useState } from "react";
import "./App.css";

function App() {
  const [method, setMethod] = useState("link");
  const [link, setLink] = useState("");
  const [links, setLinks] = useState([]);
  const [description, setDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false); // State for loading spinner
  const [results, setResults] = useState([]); // State to store the results

  const handleMethodChange = (event) => {
    setMethod(event.target.value);
  };

  const handleLinkChange = (event) => {
    setLink(event.target.value);
  };

  const handleAddLink = (event) => {
    event.preventDefault();
    if (link !== "") {
      setLinks([...links, link]);
      setLink("");
    }
  };

  const handleDeleteLink = (indexToDelete) => {
    setLinks(links.filter((_, index) => index !== indexToDelete));
  };

  const handleDescriptionChange = (event) => {
    setDescription(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true); // Show loading spinner

    if (method === "link") {
      try {
        const response = await fetch("http://localhost:5000/process_videos", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ links, description }),
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log(data);
        setResults(data); // Store the results
        console.log(data.sanitized_video_name);
      } catch (error) {
        console.error("There was a problem with the fetch operation: ", error);
      }
    }

    setIsLoading(false); // Hide loading spinner
  };

  return (
    <div className="app-container">
      <h1 className="app-title">
        ClipFinder: Search for a clip within a set of videos
      </h1>

      <form className="app-form" onSubmit={handleSubmit}>
        <div className="app-method">
          <label>
            <input
              type="radio"
              value="link"
              checked={method === "link"}
              onChange={handleMethodChange}
            />
            Link
          </label>
          <label>
            <input
              type="radio"
              value="local"
              checked={method === "local"}
              onChange={handleMethodChange}
            />
            Local
          </label>
        </div>

        {method === "link" && (
          <>
            <input
              id="videosURL"
              name="videosURL"
              type="text"
              className="app-input"
              placeholder="Videos URL"
              value={link}
              onChange={handleLinkChange}
            />
            <button onClick={handleAddLink}>Add Link</button>
            {links.map((link, index) => (
              <div key={index} className="link-item">
                <button onClick={() => handleDeleteLink(index)}>Delete</button>
                <p>{link}</p>
              </div>
            ))}
          </>
        )}

        {method === "local" && (
          <input
            id="videos"
            name="videos"
            type="file"
            className="app-input"
            multiple
          />
        )}

        <textarea
          id="clip"
          name="clip"
          className="app-input"
          rows="4"
          placeholder="Search by Keywords Seperated by commas (,) or describe what are you looking for"
          value={description}
          onChange={handleDescriptionChange}
        />

        <button type="submit" className="app-button">
          Submit
        </button>
      </form>

      {/* Loading spinner */}
      {isLoading && <div className="loading-spinner" />}

      {/* Display results */}
      <div className="results-container">
        {results.map((result, index) => (
          <div key={index}>
            <h3>{result.video_name}</h3>
            <p>Timestamp: {result.timestamp}s</p>
            <img
              src={`http://localhost:5000/Frames/visualized_frames/${result.sanitized_video_name}/${result.frame_path}`}
              alt={`Result ${result.sanitized_video_name}`} 
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
