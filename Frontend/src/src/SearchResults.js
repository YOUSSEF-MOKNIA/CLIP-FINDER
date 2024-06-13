
function SearchResults({ results }) {
    return (
      <div>
        {results.map((result, index) => (
          <div key={index}>
            <h2>Video: {result.video_name}</h2>
            <p>Timestamp: {result.timestamp}</p>
            {/* Assuming frame_path is a URL accessible by the browser */}
            <img src={result.frame_path} alt={`Frame from ${result.video_name}`} />
          </div>
        ))}
      </div>
    );
}

export default SearchResults;