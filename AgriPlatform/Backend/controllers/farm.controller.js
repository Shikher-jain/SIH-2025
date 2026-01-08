const Farm = require("../models/farm.model.js");
const ResponseEntity = require("../utils/ResponseEntity.js");
const path = require("path");
const { prepareData } = require("../utils/prepareData");
const { centroidFromRing } = require("../utils/geometry");


// Get all farms in the system (admin only)
const getAllFarms = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;

    // Get all farms with pagination
    const farms = await Farm.find({})
      .sort({ createdAt: -1 })
      .skip((page - 1) * limit)
      .limit(limit)
      .populate("userId", "fullName email");

    // Get total count for pagination
    const total = await Farm.countDocuments({});
    const totalPages = Math.ceil(total / limit);

    const response = new ResponseEntity(1, "All farms retrieved successfully", {
      farms,
      pagination: {
        page,
        limit,
        total,
        totalPages,
      },
    });
    res.status(200).json(response);
  } catch (error) {
    console.error("Error fetching all farms:", error);
    const response = new ResponseEntity(0, "Error fetching all farms", {});
    res.status(500).json(response);
  }
};
// Get all farms for authenticated user
const getFarms = async (req, res) => {
  try {
    const userId = req.user._id;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    // Get farms with pagination
    const farms = await Farm.findByUserId(userId, { page, limit });
    const total = await Farm.countDocuments({ userId });
    const totalPages = Math.ceil(total / limit);
    const response = new ResponseEntity(1, "Farms retrieved successfully", {
      farms,
      pagination: {
        page,
        limit,
        total,
        totalPages,
      },
    });

    res.status(200).json(response);
  } catch (error) {
    console.error("Error fetching farms:", error);
    const response = new ResponseEntity(0, "Error fetching farms", {});
    res.status(500).json(response);
  }
};

// Get single farm by ID
const getFarm = async (req, res) => {
  try {
    const farmId = req.params.id;
    const userId = req.user._id;

    const farm = await Farm.findById(farmId);

    if (!farm) {
      const response = new ResponseEntity(0, "Farm not found", {});
      return res.status(404).json(response);
    }

    // Check if user owns this farm
    if (!farm.isOwnedBy(userId)) {
      const response = new ResponseEntity(0, "Access denied", {});
      return res.status(403).json(response);
    }

    const response = new ResponseEntity(1, "Farm retrieved successfully", farm);
    res.status(200).json(response);
  } catch (error) {
    console.error("Error fetching farm:", error);
    const response = new ResponseEntity(0, "Error fetching farm", {});
    res.status(500).json(response);
  }
};

// Create new farm
const createFarm = async (req, res) => {
  try {
    const userId = req.user._id;

    if (!req.body) {
      const response = new ResponseEntity(0, "No request body provided", {});
      return res.status(400).json(response);
    }

    const {
      name,
      crop,
      plantingDate,
      harvestDate,
      description,
      coordinates,
      area,
    } = req.body;

    // Basic validation
    if (
      !name ||
      !crop ||
      !plantingDate ||
      !harvestDate ||
      !coordinates ||
      !area
    ) {
      const response = new ResponseEntity(0, "Missing required fields", {});
      return res.status(400).json(response);
    }

    // Validate coordinates
    if (!Array.isArray(coordinates) || coordinates.length < 3) {
      const response = new ResponseEntity(
        0,
        "Invalid coordinates. Must be an array with at least 3 points",
        {}
      );
      return res.status(400).json(response);
    }

    let closedCoordinates = [...coordinates];
    const firstPoint = coordinates[0];
    const lastPoint = coordinates[coordinates.length - 1];

    // Check if first and last points are the same
    if (firstPoint[0] !== lastPoint[0] || firstPoint[1] !== lastPoint[1]) {
      // If not, add the first point at the end to close the loop
      closedCoordinates.push([...firstPoint]);
    }

    // Validate dates
    const plantDate = new Date(plantingDate);
    const harvestDate_ = new Date(harvestDate);

    if (plantDate >= harvestDate_) {
      const response = new ResponseEntity(
        0,
        "Harvest date must be after planting date",
        {}
      );
      return res.status(400).json(response);
    }

    const geoJsonCoordinates = {
      type: "Polygon",
      coordinates: [closedCoordinates], // Wrap closed coordinates in an array for GeoJSON Polygon format
    };

    const centroid = centroidFromRing(closedCoordinates);
    console.log("CENTROID: ", centroid);
    const farmData = {
      name: name.trim(),
      crop: crop.trim(),
      plantingDate: plantDate,
      harvestDate: harvestDate_,
      description: description?.trim(),
      area: parseFloat(area),
      userId,
      coordinates: geoJsonCoordinates, // Use the GeoJSON format
    };

    const defaults = {
      satelliteConfig: {
        cloudThreshold: 50,
        dateRangeMonths: 5,
        scale: 10,
        bands: ["B2", "B3", "B4", "B5", "B8", "B11"],
        bandNames: ["Blue_B2", "Green_B3", "Red_B4", "RedEdge_B5", "NIR_B8", "SWIR_B11"],
      },
      sensorConfig: {
        scale: 1000,
        assets: {
          ECe: "projects/pk07007/assets/ECe",
          N: "projects/pk07007/assets/N",
          P: "projects/pk07007/assets/P",
          OC: "projects/pk07007/assets/OC",
          pH: "projects/pk07007/assets/pH",
        },
      },
      weatherConfig: {
        forecastDays: 3,
        fields: [
          "temperature",
          "humidity",
          "pressure",
          "windSpeed",
          "windDirection",
          "precipitation",
          "cloudCover",
        ],
      },
    };

    let dataPrep = null;
    try {
      dataPrep = await prepareData({
        areaName: name.trim(),
        polygon: closedCoordinates,
        center: centroid,
        satelliteConfig: defaults.satelliteConfig,
        sensorConfig: defaults.sensorConfig,
        weatherConfig: defaults.weatherConfig,
        tempDir: path.join(__dirname, "..", "temp"),
      });
    } catch (e) {
      dataPrep = { error: e.message };
    }

    const farm = await Farm.create(farmData);

    const response = new ResponseEntity(1, "Farm created successfully", {
      farm,
      dataPrep,
    });
    res.status(201).json(response);
  } catch (error) {
    console.error("Error creating farm:", error);

    if (error.name === "ValidationError") {
      const response = new ResponseEntity(0, error.message, {});
      return res.status(400).json(response);
    }

    const response = new ResponseEntity(0, "Error creating farm", {});
    res.status(500).json(response);
  }
};

// Update farm
const updateFarm = async (req, res) => {
  try {
    const farmId = req.params.id;
    const userId = req.user._id;
    const updateData = req.body;

    const farm = await Farm.findById(farmId);
    if (!farm) {
      return res.status(404).json(new ResponseEntity(0, "Farm not found", {}));
    }

    // Check if user owns this farm
    if (!farm.isOwnedBy(userId) && req.user.role !== "admin") {
      return res.status(403).json(new ResponseEntity(0, "Access denied", {}));
    }

    // Validate dates
    if (updateData.plantingDate && updateData.harvestDate) {
      const plantDate = new Date(updateData.plantingDate);
      const harvestDate = new Date(updateData.harvestDate);

      if (plantDate >= harvestDate) {
        return res.status(400).json(
          new ResponseEntity(0, "Harvest date must be after planting date", {})
        );
      }
    }

    // Handle coordinates
    if (updateData.coordinates) {
      const coords = updateData.coordinates.coordinates;

      if (!Array.isArray(coords) || !Array.isArray(coords[0]) || coords[0].length < 3) {
        return res.status(400).json(
          new ResponseEntity(0, "Invalid coordinates. Must be an array with at least 3 points", {})
        );
      }

      let closedCoordinates = [...coords[0]];
      const firstPoint = closedCoordinates[0];
      const lastPoint = closedCoordinates[closedCoordinates.length - 1];

      if (firstPoint[0] !== lastPoint[0] || firstPoint[1] !== lastPoint[1]) {
        closedCoordinates.push([...firstPoint]);
      }

      updateData.coordinates = {
        type: "Polygon",
        coordinates: [closedCoordinates],
      };
    }

    // Update farm
    const updatedFarm = await Farm.findByIdAndUpdate(farmId, updateData, {
      new: true,
      runValidators: true,
    });

    return res
      .status(200)
      .json(new ResponseEntity(1, "Farm updated successfully", updatedFarm));
  } catch (error) {
    console.error("Error updating farm:", error);

    if (error.name === "ValidationError") {
      return res.status(400).json(new ResponseEntity(0, error.message, {}));
    }

    return res
      .status(500)
      .json(new ResponseEntity(0, "Error updating farm", { error: error.message }));
  }
};

// Delete farm
const deleteFarm = async (req, res) => {
  try {
    const farmId = req.params.id;
    const userId = req.user._id;

    const farm = await Farm.findById(farmId);

    if (!farm) {
      const response = new ResponseEntity(0, "Farm not found", {});
      return res.status(404).json(response);
    }

    // Check if user owns this farm
    if (!farm.isOwnedBy(userId)) {
      const response = new ResponseEntity(0, "Access denied", {});
      return res.status(403).json(response);
    }

    await Farm.findByIdAndDelete(farmId);

    const response = new ResponseEntity(1, "Farm deleted successfully", {});
    res.status(200).json(response);
  } catch (error) {
    console.error("Error deleting farm:", error);
    const response = new ResponseEntity(0, "Error deleting farm", {});
    res.status(500).json(response);
  }
};

module.exports = {
  getFarms,
  getFarm,
  createFarm,
  updateFarm,
  deleteFarm,
  getAllFarms,
};
