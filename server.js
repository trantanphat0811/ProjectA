const express = require('express');
const axios = require('axios');
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post('/api/upload', async (req, res) => {
    try {
        const response = await axios.post('http://localhost:8000/detect', req.body, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/report', async (req, res) => {
    try {
        const response = await axios.get('http://localhost:8000/report');
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => console.log('Server running at http://localhost:3000'));