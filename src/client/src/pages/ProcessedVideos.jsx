import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../assets/styles/ProcessedVideos.css';

function ProcessedVideos() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');

  useEffect(() => {
    fetchProcessedVideos();
  }, []);

  const fetchProcessedVideos = async () => {
    try {
      const response = await fetch('/api/videos/processed');
      const data = await response.json();
      setVideos(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching processed videos:', error);
      setLoading(false);
    }
  };

  const handleFilterChange = (event) => {
    setFilter(event.target.value);
  };

  const handleSortChange = (event) => {
    setSortBy(event.target.value);
  };

  const handleSortOrderChange = () => {
    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
  };

  const filteredAndSortedVideos = videos
    .filter(video => 
      video.filename.toLowerCase().includes(filter.toLowerCase()) ||
      video.date.toLowerCase().includes(filter.toLowerCase())
    )
    .sort((a, b) => {
      const order = sortOrder === 'asc' ? 1 : -1;
      switch (sortBy) {
        case 'date':
          return order * (new Date(b.date) - new Date(a.date));
        case 'violations':
          return order * (b.violations - a.violations);
        case 'duration':
          return order * (b.duration - a.duration);
        default:
          return 0;
      }
    });

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="processed-videos">
      <div className="controls">
        <div className="search">
          <input
            type="text"
            placeholder="Search videos..."
            value={filter}
            onChange={handleFilterChange}
          />
        </div>
        <div className="sort">
          <select value={sortBy} onChange={handleSortChange}>
            <option value="date">Date</option>
            <option value="violations">Violations</option>
            <option value="duration">Duration</option>
          </select>
          <button onClick={handleSortOrderChange}>
            <i className={`fas fa-sort-${sortOrder === 'asc' ? 'up' : 'down'}`}></i>
          </button>
        </div>
      </div>

      <div className="videos-grid">
        {filteredAndSortedVideos.map((video) => (
          <Link to={`/video/${video.id}`} key={video.id} className="video-card">
            <div className="thumbnail">
              <img src={video.thumbnail} alt={video.filename} />
              <span className="duration">{video.duration}</span>
            </div>
            <div className="video-info">
              <h3>{video.filename}</h3>
              <p>Processed: {new Date(video.date).toLocaleDateString()}</p>
              <p>Violations: {video.violations}</p>
              <div className="stats">
                <span>
                  <i className="fas fa-car"></i>
                  {video.vehicleCount}
                </span>
                <span>
                  <i className="fas fa-tachometer-alt"></i>
                  {video.avgSpeed} km/h
                </span>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {filteredAndSortedVideos.length === 0 && (
        <div className="no-results">
          <p>No videos found</p>
        </div>
      )}
    </div>
  );
}

export default ProcessedVideos; 