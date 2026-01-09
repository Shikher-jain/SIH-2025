/**
 * JavaScript API Client for Multi-Modal CNN API
 * 
 * This provides easy integration with the Python API from JavaScript backends
 */

class MultiModalCNNClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
    }

    /**
     * Check API status and health
     */
    async getStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/predict/status`);
            return await response.json();
        } catch (error) {
            throw new Error(`API status check failed: ${error.message}`);
        }
    }

    /**
     * Check if model is loaded and ready
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/predict/health`);
            return await response.json();
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    }

    /**
     * Make prediction using JSON data (URLs)
     * 
     * @param {Object} requestData - The prediction request
     * @param {string} requestData.fieldId - Field identifier
     * @param {string} requestData.userId - User identifier  
     * @param {string} requestData.hyperSpectralImageUrl - URL to hyperspectral image
     * @param {Object} requestData.sensorData - Sensor data object
     */
    async predictWithUrls(requestData) {
        try {
            const response = await fetch(`${this.baseUrl}/predict/`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Prediction failed');
            }

            return await response.json();
        } catch (error) {
            throw new Error(`Prediction failed: ${error.message}`);
        }
    }

    /**
     * Make prediction using file uploads
     * 
     * @param {File} tifFile - TIF image file
     * @param {File} npyFile - NPY array file
     * @param {File} csvFile - CSV sensor data file
     */
    async predictWithFiles(tifFile, npyFile, csvFile) {
        try {
            const formData = new FormData();
            formData.append('tif_file', tifFile);
            formData.append('npy_file', npyFile);
            formData.append('csv_file', csvFile);

            const response = await fetch(`${this.baseUrl}/predict/files`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Prediction failed');
            }

            return await response.json();
        } catch (error) {
            throw new Error(`File prediction failed: ${error.message}`);
        }
    }

    /**
     * Get API information
     */
    async getApiInfo() {
        try {
            const response = await fetch(`${this.baseUrl}/`);
            return await response.json();
        } catch (error) {
            throw new Error(`API info request failed: ${error.message}`);
        }
    }
}

// Example usage for Node.js backend
async function exampleUsage() {
    const client = new MultiModalCNNClient('http://localhost:8000');

    try {
        // Check if API is ready
        const status = await client.getStatus();
        console.log('API Status:', status);

        // Example prediction with URLs
        const predictionRequest = {
            fieldId: "field_001",
            userId: "user_123",
            hyperSpectralImageUrl: "https://example.com/hyperspectral.npy",
            sensorData: {
                airTemperature: 25.5,
                humidity: 60.0,
                soilMoisture: 30.0,
                timestamp: new Date().toISOString(),
                windSpeed: 10.0,
                rainfall: 0.0,
                solarRadiation: 800.0,
                leafWetness: 15.0,
                co2Level: 400.0,
                phLevel: 6.5
            }
        };

        const result = await client.predictWithUrls(predictionRequest);
        console.log('Prediction Result:', result);

    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Export for use in Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MultiModalCNNClient, exampleUsage };
}

// Export for use in browsers
if (typeof window !== 'undefined') {
    window.MultiModalCNNClient = MultiModalCNNClient;
}