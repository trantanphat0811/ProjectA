import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import ProcessedVideos from './pages/ProcessedVideos';
import Violations from './pages/Violations';
import VideoDetails from './pages/VideoDetails';
import './assets/styles/App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/processed-videos" element={<ProcessedVideos />} />
            <Route path="/violations" element={<Violations />} />
            <Route path="/video/:id" element={<VideoDetails />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 