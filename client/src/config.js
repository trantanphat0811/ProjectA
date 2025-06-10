const config = {
    API_URL: process.env.REACT_APP_API_URL || 
        (process.env.NODE_ENV === 'production'
            ? 'https://your-railway-app-name.railway.app'  // Thay thế bằng URL Railway của bạn sau khi deploy
            : 'http://localhost:8000'),
    UPLOAD_MAX_SIZE: 16 * 1024 * 1024, // 16MB
    SUPPORTED_FORMATS: ['.mp4', '.avi', '.mov'],
    DEFAULT_PAGE_SIZE: 10
};

export default config; 