import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../assets/styles/Dashboard.css';

function Dashboard() {
  const [stats, setStats] = useState({
    totalVideos: 0,
    totalViolations: 0,
    recentViolations: [],
    processingVideos: []
  });

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    fetchDashboardStats();
  }, []);

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch('/api/dashboard/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    }
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('/api/videos/upload', {
        method: 'POST',
        body: formData,
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });

      if (response.ok) {
        setSelectedFile(null);
        setUploadProgress(0);
        fetchDashboardStats();
      }
    } catch (error) {
      console.error('Error uploading video:', error);
    }
  };

  return (
    <div className="dashboard">
      <section className="stats-section">
        <div className="stat-card">
          <i className="fas fa-video"></i>
          <h3>Total Videos</h3>
          <p>{stats.totalVideos}</p>
        </div>
        <div className="stat-card">
          <i className="fas fa-exclamation-triangle"></i>
          <h3>Total Violations</h3>
          <p>{stats.totalViolations}</p>
        </div>
      </section>

      <section className="upload-section">
        <h2>Upload New Video</h2>
        <div className="upload-container">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="file-input"
          />
          {selectedFile && (
            <div className="upload-progress">
              <div 
                className="progress-bar"
                style={{ width: `${uploadProgress}%` }}
              ></div>
              <button onClick={handleUpload} className="upload-button">
                Upload Video
              </button>
            </div>
          )}
        </div>
      </section>

      <section className="recent-violations">
        <h2>Recent Violations</h2>
        <div className="violations-grid">
          {stats.recentViolations.map((violation) => (
            <Link 
              to={`/violations/${violation.id}`}
              key={violation.id}
              className="violation-card"
            >
              <img src={violation.thumbnail} alt="Violation thumbnail" />
              <div className="violation-info">
                <p>Speed: {violation.speed} km/h</p>
                <p>Time: {new Date(violation.timestamp).toLocaleString()}</p>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="processing-videos">
        <h2>Processing Videos</h2>
        <div className="processing-list">
          {stats.processingVideos.map((video) => (
            <div key={video.id} className="processing-item">
              <p>{video.filename}</p>
              <div className="processing-progress">
                <div 
                  className="progress-bar"
                  style={{ width: `${video.progress}%` }}
                ></div>
                <span>{video.progress}%</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default Dashboard; 