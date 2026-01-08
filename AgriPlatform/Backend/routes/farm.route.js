const express = require("express");
const router = express.Router();
const isLoggedIn = require("../middlewares/auth.middleware");
const farmController = require("../controllers/farm.controller");

// All farm routes require authentication
router.use(isLoggedIn);

// GET /farms/all - Get all farms in the system (admin only)
router.get("/all", farmController.getAllFarms);

// GET /farms - Get all farms for authenticated user (with pagination)
router.get("/", farmController.getFarms);

// GET /farms/:id - Get specific farm by ID
router.get("/:id", farmController.getFarm);

// POST /farms - Create new farm
router.post("/", farmController.createFarm);

// PUT /farms/:id - Update farm
router.put("/:id", farmController.updateFarm);

// DELETE /farms/:id - Delete farm
router.delete("/:id", farmController.deleteFarm);

module.exports = router;
