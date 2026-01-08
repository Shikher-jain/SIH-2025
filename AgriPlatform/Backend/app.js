const express = require("express");
const cookieParser = require("cookie-parser");
const cors = require("cors");
const dotenv = require("dotenv");
const { ee } = require("./utils/geeInit");
const userRoutes = require("./routes/user.route.js");
const farmRoutes = require("./routes/farm.route.js");

const app = express();

// Middleware setup
dotenv.config();
app.use(cors({credentials: true, origin:"http://localhost:3000" }));
app.use(express.json());
app.use(cookieParser());

app.locals.ee = ee;

// Routes
app.use("/user", userRoutes);
app.use("/farms", farmRoutes);

module.exports = app;
