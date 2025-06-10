import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../assets/styles/Violations.css';

function Violations() {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    date: '',
    minSpeed: '',
    maxSpeed: '',
    vehicleType: 'all'
  });
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');

  useEffect(() => {
    fetchViolations();
  }, []);

  const fetchViolations = async () => {
    try {
      const response = await fetch('/api/violations');
      const data = await response.json();
      setViolations(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching violations:', error);
      setLoading(false);
    }
  };

  const handleFilterChange = (event) => {
    const { name, value } = event.target;
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSortChange = (event) => {
    setSortBy(event.target.value);
  };

  const handleSortOrderChange = () => {
    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
  };

  const filteredAndSortedViolations = violations
    .filter(violation => {
      const dateMatch = !filters.date || 
        violation.date.includes(filters.date);
      const speedMatch = (!filters.minSpeed || violation.speed >= Number(filters.minSpeed)) &&
        (!filters.maxSpeed || violation.speed <= Number(filters.maxSpeed));
      const typeMatch = filters.vehicleType === 'all' || 
        violation.vehicleType === filters.vehicleType;
      
      return dateMatch && speedMatch && typeMatch;
    })
    .sort((a, b) => {
      const order = sortOrder === 'asc' ? 1 : -1;
      switch (sortBy) {
        case 'date':
          return order * (new Date(b.date) - new Date(a.date));
        case 'speed':
          return order * (b.speed - a.speed);
        default:
          return 0;
      }
    });

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="violations">
      <div className="filters">
        <div className="filter-group">
          <label>Date:</label>
          <input
            type="date"
            name="date"
            value={filters.date}
            onChange={handleFilterChange}
          />
        </div>
        <div className="filter-group">
          <label>Speed Range:</label>
          <input
            type="number"
            name="minSpeed"
            placeholder="Min"
            value={filters.minSpeed}
            onChange={handleFilterChange}
          />
          <span>-</span>
          <input
            type="number"
            name="maxSpeed"
            placeholder="Max"
            value={filters.maxSpeed}
            onChange={handleFilterChange}
          />
        </div>
        <div className="filter-group">
          <label>Vehicle Type:</label>
          <select
            name="vehicleType"
            value={filters.vehicleType}
            onChange={handleFilterChange}
          >
            <option value="all">All</option>
            <option value="car">Car</option>
            <option value="truck">Truck</option>
            <option value="motorcycle">Motorcycle</option>
            <option value="bus">Bus</option>
          </select>
        </div>
        <div className="sort-group">
          <select value={sortBy} onChange={handleSortChange}>
            <option value="date">Date</option>
            <option value="speed">Speed</option>
          </select>
          <button onClick={handleSortOrderChange}>
            <i className={`fas fa-sort-${sortOrder === 'asc' ? 'up' : 'down'}`}></i>
          </button>
        </div>
      </div>

      <div className="violations-grid">
        {filteredAndSortedViolations.map((violation) => (
          <div key={violation.id} className="violation-card">
            <div className="violation-image">
              <img src={violation.image} alt="Violation" />
              <span className="speed">{violation.speed} km/h</span>
            </div>
            <div className="violation-info">
              <p className="date">{new Date(violation.date).toLocaleString()}</p>
              <p className="vehicle-type">
                <i className={`fas fa-${violation.vehicleType === 'motorcycle' ? 'motorcycle' : 'car'}`}></i>
                {violation.vehicleType}
              </p>
              <p className="location">Location: {violation.location}</p>
              <Link to={`/video/${violation.videoId}`} className="video-link">
                View Video
              </Link>
            </div>
          </div>
        ))}
      </div>

      {filteredAndSortedViolations.length === 0 && (
        <div className="no-results">
          <p>No violations found</p>
        </div>
      )}
    </div>
  );
}

export default Violations; 