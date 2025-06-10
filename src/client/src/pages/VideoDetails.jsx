import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import '../assets/styles/VideoDetails.css';

function VideoDetails() {
  const { id } = useParams();
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [speedData, setSpeedData] = useState({
    labels: [],
    datasets: []
  });

  useEffect(() => {
    fetchVideoDetails();
  }, [id]);

  const fetchVideoDetails = async () => {
    try {
      const response = await fetch(`/api/videos/${id}`);
      const data = await response.json();
      setVideo(data);
      
      // Prepare speed data for chart
      const speedDataset = {
        labels: data.speedData.map(d => d.time),
        datasets: [{
          label: 'Vehicle Speed',
          data: data.speedData.map(d => d.speed),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }]
      };
      setSpeedData(speedDataset);
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching video details:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (!video) {
    return <div className="error">Video not found</div>;
  }

  return (
    <div className="video-details">
      <div className="video-header">
        <h1>{video.filename}</h1>
        <div className="video-meta">
          <span>
            <i className="fas fa-calendar"></i>
            {new Date(video.date).toLocaleDateString()}
          </span>
          <span>
            <i className="fas fa-clock"></i>
            {video.duration}
          </span>
          <span>
            <i className="fas fa-exclamation-triangle"></i>
            {video.violations} violations
          </span>
        </div>
      </div>

      <div className="video-content">
        <div className="video-player">
          <video controls src={video.url}>
            Your browser does not support the video tag.
          </video>
        </div>

        <div className="tabs">
          <button
            className={activeTab === 'overview' ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={activeTab === 'violations' ? 'active' : ''}
            onClick={() => setActiveTab('violations')}
          >
            Violations
          </button>
          <button
            className={activeTab === 'statistics' ? 'active' : ''}
            onClick={() => setActiveTab('statistics')}
          >
            Statistics
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'overview' && (
            <div className="overview">
              <div className="stats-grid">
                <div className="stat-card">
                  <h3>Total Vehicles</h3>
                  <p>{video.vehicleCount}</p>
                </div>
                <div className="stat-card">
                  <h3>Average Speed</h3>
                  <p>{video.avgSpeed} km/h</p>
                </div>
                <div className="stat-card">
                  <h3>Max Speed</h3>
                  <p>{video.maxSpeed} km/h</p>
                </div>
                <div className="stat-card">
                  <h3>Violation Rate</h3>
                  <p>{((video.violations / video.vehicleCount) * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="speed-chart">
                <h3>Speed Distribution</h3>
                <Line data={speedData} options={{
                  responsive: true,
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'Speed (km/h)'
                      }
                    },
                    x: {
                      title: {
                        display: true,
                        text: 'Time'
                      }
                    }
                  }
                }} />
              </div>
            </div>
          )}

          {activeTab === 'violations' && (
            <div className="violations-list">
              {video.violations.map((violation) => (
                <div key={violation.id} className="violation-item">
                  <div className="violation-image">
                    <img src={violation.image} alt="Violation" />
                  </div>
                  <div className="violation-details">
                    <p className="speed">{violation.speed} km/h</p>
                    <p className="time">{new Date(violation.timestamp).toLocaleString()}</p>
                    <p className="vehicle-type">{violation.vehicleType}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'statistics' && (
            <div className="statistics">
              <div className="vehicle-types">
                <h3>Vehicle Type Distribution</h3>
                <div className="vehicle-type-chart">
                  {/* Add pie chart for vehicle types */}
                </div>
              </div>
              <div className="hourly-distribution">
                <h3>Hourly Traffic Distribution</h3>
                <div className="hourly-chart">
                  {/* Add bar chart for hourly distribution */}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="actions">
        <button className="download-btn">
          <i className="fas fa-download"></i>
          Download Report
        </button>
        <button className="share-btn">
          <i className="fas fa-share"></i>
          Share
        </button>
      </div>
    </div>
  );
}

export default VideoDetails; 