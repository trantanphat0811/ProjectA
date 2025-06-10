import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import '../assets/styles/Navbar.css';

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">
          <h1>Traffic Speed Detection</h1>
        </Link>
      </div>
      <ul className="nav-links">
        <li className={location.pathname === '/' ? 'active' : ''}>
          <Link to="/">
            <i className="fas fa-home"></i>
            Dashboard
          </Link>
        </li>
        <li className={location.pathname === '/processed-videos' ? 'active' : ''}>
          <Link to="/processed-videos">
            <i className="fas fa-video"></i>
            Processed Videos
          </Link>
        </li>
        <li className={location.pathname === '/violations' ? 'active' : ''}>
          <Link to="/violations">
            <i className="fas fa-exclamation-triangle"></i>
            Violations
          </Link>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar; 