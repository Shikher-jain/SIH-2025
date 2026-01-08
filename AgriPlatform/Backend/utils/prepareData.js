const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const axios = require('axios');
const { ee } = require('./geeInit');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function safeName(name) {
  return name.replace(/[^a-zA-Z0-9_-]+/g, '_');
}

async function downloadToFile(url, filepath) {
  const res = await axios.get(url, { responseType: 'arraybuffer', maxBodyLength: Infinity, maxContentLength: Infinity });
  fs.writeFileSync(filepath, res.data);
  const stats = fs.statSync(filepath);
  return { sizeMB: stats.size / (1024 * 1024) };
}

function dateRangeMonths(months) {
  const end = new Date();
  const start = new Date(end);
  start.setMonth(start.getMonth() - months);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

function getCentroidFromRing(ring) {
  let twiceArea = 0, x = 0, y = 0;
  for (let i = 0, l = ring.length - 1; i < l; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[i + 1];
    const f = x1 * y2 - x2 * y1;
    twiceArea += f;
    x += (x1 + x2) * f;
    y += (y1 + y2) * f;
  }
  if (twiceArea === 0) {
    const sx = ring.reduce((a, p) => a + p[0], 0) / ring.length;
    const sy = ring.reduce((a, p) => a + p[1], 0) / ring.length;
    return [sx, sy];
  }
  const area = twiceArea * 0.5;
  return [x / (6 * area), y / (6 * area)];
}

async function getImageInfo(image) {
  return await new Promise((resolve, reject) => {
    image.getInfo((info, err) => {
      if (err) reject(new Error(String(err)));
      else resolve(info);
    });
  });
}

async function getDownloadURL(image, params) {
  return await new Promise((resolve, reject) => {
    try {
      image.getDownloadURL(params, (url, err) => {
        if (err) return reject(new Error(String(err)));
        if (!url) return reject(new Error('No download URL generated'));
        resolve(url);
      });
    } catch (e) {
      reject(e);
    }
  });
}

async function satelliteTask(areaName, ring, cfg, jobDir) {
  try {
    const region = ee.Geometry.Polygon([ring]);
    const range = dateRangeMonths(cfg.dateRangeMonths);
    const collection = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(region)
      .filterDate(range.start, range.end)
      .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cfg.cloudThreshold))
      .sort('system:time_start', false);
    const image = collection.first().clip(region).select(cfg.bands).rename(cfg.bandNames);
    const info = await getImageInfo(image);
    const ts = info && info.properties && info.properties['system:time_start'] ? new Date(info.properties['system:time_start']).toISOString().slice(0,10) : new Date().toISOString().slice(0,10);
    const filename = `${safeName(areaName)}_satellite_${ts}_${Date.now()}.tif`;
    const filepath = path.join(jobDir, filename);
    const url = await getDownloadURL(image, { region: region, scale: cfg.scale, format: 'GEO_TIFF', maxPixels: 1e10 });
    const { sizeMB } = await downloadToFile(url, filepath);
    console.log("Satellite Task Completed")
    return { success: true, filepath, sizeMB, imageDate: ts, bands: cfg.bandNames, scale: cfg.scale, dateRange: range };
  } catch (e) {
    return null;
  }
}

async function sensorTask(areaName, ring, cfg, jobDir) {
  try {
    const region = ee.Geometry.Polygon([ring]);
    const loaded = [];
    const images = [];
    for (const [name, asset] of Object.entries(cfg.assets)) {
      try {
        const img = ee.Image(asset);
        const info = await getImageInfo(img);
        const bandNames = (info.bands || []).map((b, i) => `${name}_${b.id || `Layer${i + 1}`}`);
        images.push(img.rename(bandNames));
        loaded.push({ name, bandCount: bandNames.length, bandNames });
      } catch (_) {}
    }
    if (images.length === 0) return null;
    const combined = ee.Image.cat(images).clip(region);
    const filename = `${safeName(areaName)}_sensor_${Date.now()}.tif`;
    const filepath = path.join(jobDir, filename);
    const url = await getDownloadURL(combined, { region: region, scale: cfg.scale, format: 'GEO_TIFF', maxPixels: 1e10 });
    const { sizeMB } = await downloadToFile(url, filepath);
    const totalBands = loaded.reduce((a, s) => a + s.bandCount, 0);
    const allBandNames = loaded.flatMap(s => s.bandNames);
    console.log("Sensor Task Completed")
    return { success: true, filepath, sizeMB, sensorsLoaded: loaded.map(s => s.name), totalBands, bands: allBandNames, scale: cfg.scale };
  } catch (e) {
    return null;
  }
}

async function weatherTask(areaName, center, cfg, jobDir) {
  try {
    const [lon, lat] = center;
    const apiKey = cfg.apiKey || process.env.OPENWEATHER_API_KEY;
    if (!apiKey) throw new Error('Missing OpenWeather API key');
    const base = 'https://api.openweathermap.org/data/2.5';
    const current = await axios.get(`${base}/weather`, { params: { lat, lon, appid: apiKey, units: 'metric' } });
    const forecast = await axios.get(`${base}/forecast`, { params: { lat, lon, appid: apiKey, units: 'metric' } });
    const days = cfg.forecastDays || 3;
    const byDay = {};
    for (const item of forecast.data.list || []) {
      const d = item.dt_txt.slice(0, 10);
      if (!byDay[d]) byDay[d] = [];
      byDay[d].push(item);
    }
    const daily = Object.keys(byDay).slice(0, days).map(d => {
      const arr = byDay[d];
      const temps = arr.map(x => x.main.temp);
      const humidity = arr.map(x => x.main.humidity);
      const pressure = arr.map(x => x.main.pressure);
      const wind = arr.map(x => x.wind.speed);
      return {
        date: d,
        temperature: { min: Math.min(...temps), max: Math.max(...temps), avg: temps.reduce((a, b) => a + b, 0) / temps.length },
        humidity: { avg: humidity.reduce((a, b) => a + b, 0) / humidity.length },
        pressure: { avg: pressure.reduce((a, b) => a + b, 0) / pressure.length },
        windSpeed: { avg: wind.reduce((a, b) => a + b, 0) / wind.length },
        precipitation: (arr.reduce((a, x) => a + ((x.rain && x.rain['3h']) || 0), 0))
      };
    });
    const processed = { center, current: current.data, daily, fields: cfg.fields };
    const filename = `${safeName(areaName)}_weather_${Date.now()}.json`;
    const filepath = path.join(jobDir, filename);
    fs.writeFileSync(filepath, JSON.stringify(processed));
    const stats = fs.statSync(filepath);
    console.log("Weather Task Completed");
    return { success: true, filepath, sizeMB: stats.size / (1024 * 1024), dataPoints: (forecast.data.list || []).length };
  } catch (e) {
    console.log("Weather Task is Fucked: ", e);
    return null;
  }
}

function calcBounds(ring) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of ring) { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y; }
  return [minX, minY, maxX, maxY];
}

async function prepareData(options) {
  const areaName = options.areaName;
  const ring = options.polygon;
  const center = options.center || getCentroidFromRing(ring);
  const baseTemp = options.tempDir || path.join(process.cwd(), 'Backend', 'temp');
  const id = crypto.randomBytes(4).toString('hex');
  const stamp = new Date().toISOString().replace(/[:.]/g, '').slice(0, 15);
  const jobDir = path.join(baseTemp, `${safeName(areaName)}_${stamp}_${id}`);
  ensureDir(jobDir);

  const satelliteConfig = Object.assign({ cloudThreshold: 50, dateRangeMonths: 5, scale: 10, bands: ['B2', 'B3', 'B4', 'B5', 'B8', 'B11'], bandNames: ['Blue_B2', 'Green_B3', 'Red_B4', 'RedEdge_B5', 'NIR_B8', 'SWIR_B11'] }, options.satelliteConfig || {});
  const sensorConfig = Object.assign({ scale: 1000, assets: { ECe: 'projects/pk07007/assets/ECe', N: 'projects/pk07007/assets/N', P: 'projects/pk07007/assets/P', OC: 'projects/pk07007/assets/OC', pH: 'projects/pk07007/assets/pH' } }, options.sensorConfig || {});
  const weatherConfig = Object.assign({ forecastDays: 3, fields: ['temperature', 'humidity', 'pressure', 'windSpeed', 'windDirection', 'precipitation', 'cloudCover'] }, options.weatherConfig || {});

  const t0 = Date.now();
  const [satellite, sensor, weather] = await Promise.all([
    satelliteTask(areaName, ring, satelliteConfig, jobDir),
    sensorTask(areaName, ring, sensorConfig, jobDir),
    weatherTask(areaName, center, weatherConfig, jobDir)
  ]);

  if (!satellite && !sensor && !weather) throw new Error('All data preparation tasks failed');
  const metadata = {
    areaName,
    generation_date: new Date().toISOString(),
    geometry: { bounds: calcBounds(ring), center, polygon: ring },
    files: {
      satellite: satellite ? { path: satellite.filepath, sizeMB: satellite.sizeMB } : null,
      sensor: sensor ? { path: sensor.filepath, sizeMB: sensor.sizeMB } : null,
      weather: weather ? { path: weather.filepath, sizeMB: weather.sizeMB } : null
    },
    satellite: satellite ? { bands: satellite.bands, scale: satellite.scale, dateRange: satellite.dateRange, imageDate: satellite.imageDate } : null,
    sensor: sensor ? { sensors: sensor.sensorsLoaded, totalBands: sensor.totalBands, bands: sensor.bands, scale: sensor.scale } : null,
    weather: weather ? { fields: weatherConfig.fields } : null,
    quality: { satellite: !!satellite, sensor: !!sensor, weather: !!weather },
    processing_ms: Date.now() - t0
  };

  const metadataPath = path.join(jobDir, 'metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata));
  console.log("Metadata Prepared");
  return {
    tempDir: jobDir,
    files: {
      satellitePath: satellite ? satellite.filepath : null,
      sensorPath: sensor ? sensor.filepath : null,
      weatherPath: weather ? weather.filepath : null
    },
    metadataPath,
    summary: { satellite: !!satellite, sensor: !!sensor, weather: !!weather }
  };
}

async function cleanupData(targetPath) {
  if (!fs.existsSync(targetPath)) return;
  const stat = fs.statSync(targetPath);
  if (stat.isDirectory()) {
    fs.rmSync(targetPath, { recursive: true, force: true });
  } else {
    fs.unlinkSync(targetPath);
  }
}

module.exports = { prepareData, cleanupData };
