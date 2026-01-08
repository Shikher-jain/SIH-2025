import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMapEvents, Polygon, ScaleControl, ZoomControl } from 'react-leaflet';
import L, {type LeafletMouseEvent } from 'leaflet';
import { Square, Pentagon, Check, Trash2, ZoomIn, ZoomOut, Layers, RotateCcw, LocateFixed } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import './map.css';
import { formatHectares } from '@/utils';

// Fix for default markers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface LeafletMapProps {
  onPolygonComplete?: (coordinates: number[][], area: number) => void;
  initialCoordinates?: number[][];
  height?: string;
  className?: string;
}

// Maximum allowed area in hectares (100 km² = 10,000 hectares)
const MAX_AREA_HECTARES = parseInt(import.meta.env.VITE_MAX_AREA_HECTARES  || "10000");
interface DrawingControlsProps {
  onPolygonComplete: ((coordinates: number[][], area: number) => void) | undefined;
  isDrawing: boolean;
  setIsDrawing: (drawing: boolean) => void;
  currentPolygon: [number, number][];
  setCurrentPolygon: (polygon: [number, number][]) => void;
  drawingMode: 'polygon' | 'rectangle' | null;
  setDrawingMode: (mode: 'polygon' | 'rectangle' | null) => void;
}

const DrawingControls: React.FC<DrawingControlsProps> = ({
  onPolygonComplete,
  isDrawing,
  setIsDrawing,
  currentPolygon,
  setCurrentPolygon,
  drawingMode,
  setDrawingMode
}) => {
  const [rectangleStart, setRectangleStart] = useState<[number, number] | null>(null);

  useMapEvents({
    click: (e: LeafletMouseEvent) => {
      if (!isDrawing) return;

      const newPoint: [number, number] = [e.latlng.lat, e.latlng.lng];

      if (drawingMode === 'rectangle') {
        if (!rectangleStart) {
          setRectangleStart(newPoint);
          setCurrentPolygon([newPoint]);
        } else {
          // Complete rectangle
          const [lat1, lng1] = rectangleStart;
          const [lat2, lng2] = newPoint;
          const rectanglePolygon: [number, number][] = [
            [lat1, lng1],
            [lat1, lng2],
            [lat2, lng2],
            [lat2, lng1],
            [lat1, lng1] // Close the rectangle
          ];
          
          // Check area limit before completing
          const area = calculatePolygonArea(rectanglePolygon);
          if (area > MAX_AREA_HECTARES) {
            alert(`Area too large! Maximum allowed: ${MAX_AREA_HECTARES.toLocaleString()} hectares (${area.toLocaleString()} hectares selected)`);
            setIsDrawing(false);
            setDrawingMode(null);
            setRectangleStart(null);
            setCurrentPolygon([]);
            return;
          }
          
          setCurrentPolygon(rectanglePolygon);
          setIsDrawing(false);
          setDrawingMode(null);
          setRectangleStart(null);

          const coordinates = rectanglePolygon.map(point => [point[1], point[0]]);
          if (onPolygonComplete) {
            onPolygonComplete(coordinates, area);
          }
        }
      } else if (drawingMode === 'polygon') {
        const newPolygon = [...currentPolygon, newPoint];
        setCurrentPolygon(newPolygon);
      }
    },
    dblclick: (e: LeafletMouseEvent) => {
      if (!isDrawing || drawingMode !== 'polygon' || currentPolygon.length < 3) return;

      e.originalEvent.preventDefault();
      
      // Check area limit before completing
      const area = calculatePolygonArea(currentPolygon);
      if (area > MAX_AREA_HECTARES) {
        alert(`Area too large! Maximum allowed: ${MAX_AREA_HECTARES.toLocaleString()} hectares (${area.toLocaleString()} hectares selected)`);
        setIsDrawing(false);
        setDrawingMode(null);
        setCurrentPolygon([]);
        return;
      }
      
      setIsDrawing(false);
      setDrawingMode(null);

      // Calculate area and notify parent
      const coordinates = currentPolygon.map(point => [point[1], point[0]]); // Convert to [lng, lat]
      if (onPolygonComplete) {
        onPolygonComplete(coordinates, area);
      }
    },
    mousemove: (e: LeafletMouseEvent) => {
      if (!isDrawing || drawingMode !== 'rectangle' || !rectangleStart) return;

      const currentPoint: [number, number] = [e.latlng.lat, e.latlng.lng];
      const [lat1, lng1] = rectangleStart;
      const [lat2, lng2] = currentPoint;

      const rectanglePolygon: [number, number][] = [
        [lat1, lng1],
        [lat1, lng2],
        [lat2, lng2],
        [lat2, lng1],
        [lat1, lng1]
      ];
      setCurrentPolygon(rectanglePolygon);
    }
  });

  return null;
};

const calculatePolygonArea = (coordinates: [number, number][]): number => {
  if (coordinates.length < 3) return 0;

  let area = 0;
  const n = coordinates.length;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const currentPoint = coordinates[i];
    const nextPoint = coordinates[j];

    if (currentPoint && nextPoint) {
      area += currentPoint[1] * nextPoint[0]; // lng * lat
      area -= nextPoint[1] * currentPoint[0]; // lng * lat
    }
  }

  area = Math.abs(area) / 2;

  // Convert from square degrees to hectares (approximate)
  const hectares = area * 111320 * 111320 / 10000;
  return Math.round(hectares * 100) / 100;
};

export const LeafletMap: React.FC<LeafletMapProps> = ({
  onPolygonComplete,
  initialCoordinates = [],
  height = '400px',
  className = ''
}) => {
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPolygon, setCurrentPolygon] = useState<[number, number][]>([]);
  const [drawingMode, setDrawingMode] = useState<'polygon' | 'rectangle' | null>(null);
  const [mapStyle, setMapStyle] = useState<'hybrid' | 'satellite' | 'streets'>('hybrid');
  const [showStyleSelector, setShowStyleSelector] = useState(false);
  const mapRef = useRef<L.Map | null>(null);
  const MAP_API_KEY = import.meta.env.VITE_MAP_API_KEY;

  // Convert initial coordinates if provided
  useEffect(() => {
    if (initialCoordinates.length > 0) {
      const leafletCoords: [number, number][] = initialCoordinates
        .filter(coord => coord.length >= 2 && typeof coord[0] === 'number' && typeof coord[1] === 'number')
        .map(coord => [coord[1] as number, coord[0] as number]);
      setCurrentPolygon(leafletCoords);
    }
  }, [initialCoordinates]);

  const handleStartDrawing = (mode: 'polygon' | 'rectangle') => {
    setIsDrawing(true);
    setDrawingMode(mode);
    setCurrentPolygon([]);
    setShowStyleSelector(false);
  };

  const handleClearDrawing = () => {
    setIsDrawing(false);
    setDrawingMode(null);
    setCurrentPolygon([]);
  };

  const handleFinishDrawing = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    if (currentPolygon.length < 3) {
      alert('Please add at least 3 points to create a polygon');
      return;
    }

    // Check area limit before completing
    const area = calculatePolygonArea(currentPolygon);
    if (area > MAX_AREA_HECTARES) {
      alert(`Area too large! Maximum allowed: ${MAX_AREA_HECTARES.toLocaleString()} hectares (${area.toLocaleString()} hectares selected)`);
      handleClearDrawing();
      return;
    }

    setIsDrawing(false);
    setDrawingMode(null);
    const coordinates = currentPolygon.map(point => [point[1], point[0]]); // Convert to [lng, lat]
    if (onPolygonComplete) {
      onPolygonComplete(coordinates, area);
    }
  };

  const handleZoomIn = () => {
    if (mapRef.current) {
      mapRef.current.zoomIn();
    }
  };

  const handleZoomOut = () => {
    if (mapRef.current) {
      mapRef.current.zoomOut();
    }
  };

  const handleResetView = () => {
    if (mapRef.current) {
      mapRef.current.setView(defaultCenter, Number(import.meta.env.VITE_MAP_DEFAULT_ZOOM) || 10);
    }
  };

 const handleLocateMe = () => {
  const map = mapRef.current;
  if (!map) return;

  if (!navigator.geolocation) return;

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const { latitude, longitude } = pos.coords;
      map.flyTo([latitude, longitude], 16, {
        animate: true,
        duration: 2 // seconds
      });
    },
    () => {
      // Ignore errors silently
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
};


  const handleStyleChange = (style: 'hybrid' | 'satellite' | 'streets') => {
    setMapStyle(style);
    setShowStyleSelector(false);
  };

  // Map tile URLs for different styles
  const mapStyles = {
    hybrid: `https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.jpg?key=${MAP_API_KEY}`,
    satellite: `https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key=${MAP_API_KEY}`,
    streets: `https://api.maptiler.com/maps/streets-v2/{z}/{x}/{y}.png?key=${MAP_API_KEY}`,
  };

  const defaultCenter: [number, number] = [
    Number(import.meta.env.VITE_MAP_DEFAULT_CENTER_LAT) || 28.6139,
    Number(import.meta.env.VITE_MAP_DEFAULT_CENTER_LNG) || 77.2090
  ];

  return (
    <div className={`relative ${className}`} style={{ zIndex: 1 }}>
      {/* Map Container */}
      <div style={{ height, zIndex: 1 }} className="w-full rounded-lg overflow-hidden border border-neutral-200">
        <MapContainer
          center={defaultCenter}
          zoom={Number(import.meta.env.VITE_MAP_DEFAULT_ZOOM) || 10}
          style={{ height: '100%', width: '100%' }}
          zoomControl={false}
          ref={mapRef}
        >
          {/* Base Layer - Dynamic based on selected style */}
          <TileLayer
            key={mapStyle}
            url={mapStyles[mapStyle]}
            attribution='&copy; <a href="https://www.maptiler.com/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
          />

          {/* Scale Control - No zoom control */}
          <ScaleControl position="bottomleft" imperial={false} />

          {/* Drawing Controls */}
          <DrawingControls
            onPolygonComplete={onPolygonComplete}
            isDrawing={isDrawing}
            setIsDrawing={setIsDrawing}
            currentPolygon={currentPolygon}
            setCurrentPolygon={setCurrentPolygon}
            drawingMode={drawingMode}
            setDrawingMode={setDrawingMode}
          />

          {/* Display current polygon */}
          {currentPolygon.length > 1 && (
            <Polygon
              positions={currentPolygon}
              pathOptions={{
                color: (() => {
                  const area = calculatePolygonArea(currentPolygon);
                  if (area > MAX_AREA_HECTARES) return '#ef4444'; // Red for over limit
                  if (area > MAX_AREA_HECTARES * 0.8) return '#f59e0b'; // Orange for near limit
                  return '#3b82f6'; // Blue for normal
                })(),
                weight: 2,
                fillOpacity: (() => {
                  const area = calculatePolygonArea(currentPolygon);
                  if (area > MAX_AREA_HECTARES) return 0.4; // More visible when over limit
                  return 0.2;
                })(),
                fillColor: (() => {
                  const area = calculatePolygonArea(currentPolygon);
                  if (area > MAX_AREA_HECTARES) return '#ef4444';
                  if (area > MAX_AREA_HECTARES * 0.8) return '#f59e0b';
                  return '#3b82f6';
                })()
              }}
            />
          )}
        </MapContainer>
      </div>

      {/* Left Side Controls Panel */}
      <div className="absolute top-3 left-3 z-[1000] bg-white rounded-md shadow-lg border border-neutral-700">
        <div className="p-2 space-y-1.5">
          {!isDrawing ? (
            <>
              {/* Drawing Tools */}
              <button
                type="button"
                onClick={() => handleStartDrawing('polygon')}
                className="w-8 h-8 bg-white text-black rounded-sm hover:bg-gray-100 flex items-center justify-center transition-all duration-200 group"
                title="Draw Polygon"
              >
                <Pentagon className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>
              <button
                type="button"
                onClick={() => handleStartDrawing('rectangle')}
                className="w-8 h-8 bg-white text-black rounded-sm hover:bg-gray-100 flex items-center justify-center transition-all duration-200 group"
                title="Draw Rectangle"
              >
                <Square className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>


              {/* Navigation Tools */}
              <button
                type="button"
                onClick={handleZoomIn}
                className="w-8 h-8 bg-white text-black rounded-sm hover:bg-gray-100 flex items-center justify-center transition-all duration-200 group"
                title="Zoom In"
              >
                <ZoomIn className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>
              <button
                type="button"
                onClick={handleZoomOut}
                className="w-8 h-8 bg-white text-black rounded-sm hover:bg-gray-100 flex items-center justify-center transition-all duration-200 group"
                title="Zoom Out"
              >
                <ZoomOut className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>
              <button
                type="button"
                onClick={handleResetView}
                className="w-8 h-8 bg-white text-black rounded-sm hover:bg-gray-100 flex items-center justify-center transition-all duration-200 group"
                title="Reset View"
              >
                <RotateCcw className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>
              <button
                type="button"
                onClick={handleLocateMe}
                className="w-8 h-8 bg-white text-black rounded-sm hover:bg-gray-100 flex items-center justify-center transition-all duration-200 group"
                title="My Location"
              >
                <LocateFixed className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>

              {/* Map Style Toggle */}
              <button
                type="button"
                onClick={() => setShowStyleSelector(!showStyleSelector)}
                className={`w-8 h-8 rounded-sm flex items-center justify-center transition-all duration-200 group ${
                  showStyleSelector 
                    ? 'bg-gray-100 text-black' 
                    : 'bg-white text-black hover:bg-gray-100'
                }`}
                title="Map Style"
              >
                <Layers className="h-4 w-4 group-hover:scale-110 transition-transform" />
              </button>
            </>
          ) : (
            <>
              {/* Drawing Status */}
              <div className="text-xs text-gray-700 text-center py-1 px-2 bg-gray-200 rounded-sm">
                {drawingMode} ({currentPolygon.length})
              </div>

              {/* Drawing Actions */}
              {drawingMode === 'polygon' && (
                <button
                  type="button"
                  onClick={handleFinishDrawing}
                  disabled={currentPolygon.length < 3}
                  className="w-full px-2 py-1.5 bg-emerald-600 text-white rounded-sm hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-1 transition-all duration-200 text-xs font-medium"
                  title="Finish Drawing"
                >
                  <Check className="h-3 w-3" />
                  <span>Finish</span>
                </button>
              )}
              <button
                type="button"
                onClick={handleClearDrawing}
                className="w-full px-2 py-1.5 bg-red-600 text-white rounded-sm hover:bg-red-500 flex items-center justify-center space-x-1 transition-all duration-200 text-xs font-medium"
                title="Clear Drawing"
              >
                <Trash2 className="h-3 w-3" />
                <span>Clear</span>
              </button>
            </>
          )}
        </div>

        {/* Map Style Selector */}
        {showStyleSelector && !isDrawing && (
          <div className="absolute left-full top-0 ml-2 bg-white rounded-md shadow-lg border border-neutral-700 py-1 min-w-[100px]">
            <button
              type="button"
              onClick={() => handleStyleChange('hybrid')}
              className={`w-full px-3 py-1.5 text-xs text-left transition-colors ${
                mapStyle === 'hybrid' 
                  ? 'bg-gray-100 text-black' 
                  : 'bg-white text-black hover:bg-gray-100'
              }`}
            >
              Hybrid
            </button>
            <button
              type="button"
              onClick={() => handleStyleChange('satellite')}
              className={`w-full px-3 py-1.5 text-xs text-left transition-colors ${
                mapStyle === 'satellite' 
                  ? 'bg-gray-100 text-black' 
                  : 'bg-white text-black hover:bg-gray-100'
              }`}
            >
              Satellite
            </button>
            <button
              type="button"
              onClick={() => handleStyleChange('streets')}
              className={`w-full px-3 py-1.5 text-xs text-left transition-colors ${
                mapStyle === 'streets' 
                  ? 'bg-gray-100 text-black' 
                  : 'bg-white text-black hover:bg-gray-100'
              }`}
            >
              Streets
            </button>
          </div>
        )}
      </div>

      {/* Status Display */}
      {currentPolygon.length > 0 && (
        <div className="absolute bottom-3 right-3 bg-white text-black px-3 py-2 rounded-md shadow-lg border border-neutral-700 text-sm z-[1000]">
          <div className="font-medium text-black">Points: {currentPolygon.length}</div>
          {currentPolygon.length > 2 && (() => {
            const area = calculatePolygonArea(currentPolygon);
            const isOverLimit = area > MAX_AREA_HECTARES;
            const isNearLimit = area > MAX_AREA_HECTARES * 0.8;
            
            return (
              <>
                <div className={`font-medium ${
                  isOverLimit ? 'text-red-400' : isNearLimit ? 'text-orange-400' : 'text-emerald-400'
                }`}>
                  Area: {formatHectares(area)} ha
                </div>
                {isOverLimit && (
                  <div className="text-red-400 text-xs mt-1 font-medium">
                    ⚠️ Exceeds limit ({MAX_AREA_HECTARES.toLocaleString()} ha max)
                  </div>
                )}
                {isNearLimit && !isOverLimit && (
                  <div className="text-orange-400 text-xs mt-1">
                    ⚠️ Approaching limit ({MAX_AREA_HECTARES.toLocaleString()} ha max)
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}

      {/* Instructions */}
      {isDrawing && (
        <div className="absolute top-3 right-3 bg-white text-black px-4 py-3 rounded-md max-w-xs z-[1000] shadow-lg border border-neutral-700">
          <div className="font-medium mb-2 flex items-center text-black">
            {drawingMode === 'polygon' ? (
              <Pentagon className="h-4 w-4 mr-2 text-emerald-400" />
            ) : (
              <Square className="h-4 w-4 mr-2 text-emerald-400" />
            )}
            Drawing {drawingMode}
          </div>
          <div className="text-xs leading-relaxed text-black">
            {drawingMode === 'polygon' ? (
              <>
                • Click to add points<br />
                • Need at least 3 points<br />
                • Double-click to finish<br />
                • Or use "Finish" button
              </>
            ) : (
              <>
                • Click first corner<br />
                • Click opposite corner<br />
                • Rectangle created automatically
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};