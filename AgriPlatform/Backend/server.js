const mongoose = require("mongoose");
const dotenv = require("dotenv");
const app = require("./app");
const { initializeEE } = require("./utils/geeInit");

dotenv.config();

const PORT = process.env.PORT || 8000;

// MongoDB connection
async function startServer() {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB");
    
    await initializeEE();
    
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}

startServer();
