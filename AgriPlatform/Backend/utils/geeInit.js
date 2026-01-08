const ee = require('@google/earthengine');
const fs = require('fs');
const path = require('path');

function initializeEE() {
  return new Promise((resolve, reject) => {
    try {
      const privateKeyPath = path.join(__dirname, '..', 'earth-engine-service-account.json');
      if (!fs.existsSync(privateKeyPath)) {
        throw new Error(`Service account key not found at: ${privateKeyPath}`);
      }
      const privateKey = JSON.parse(fs.readFileSync(privateKeyPath, 'utf8'));
      console.log('🔑 Authenticating with Google Earth Engine...');
      ee.data.authenticateViaPrivateKey(privateKey, (error) => {
        if (error) {
          reject(new Error(`Authentication failed: ${error}`));
          return;
        }
        ee.initialize(null, null, (initError) => {
          if (initError) {
            reject(new Error(`Initialization failed: ${initError}`));
            return;
          }
          console.log('🌍 Google Earth Engine initialized successfully!');
          resolve();
        });
      });
    } catch (error) {
      reject(error);
    }
  });
}

module.exports = { initializeEE, ee };
