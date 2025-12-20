import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

interface GeoPoint {
  lat: number;
  lon: number;
  count: number;
}

interface GeoMapProps {
  points: GeoPoint[];
}

export function GeoMap({ points }: GeoMapProps) {
  if (!points || points.length === 0) {
    return (
      <div className="bg-muted text-muted-foreground p-4 rounded-lg">
        No geographic data available for this query.
      </div>
    );
  }

  const center: [number, number] = [points[0].lat, points[0].lon];

  return (
    <div className="w-full h-[450px] rounded-lg overflow-hidden border">
      <MapContainer
        center={center}
        zoom={3}
        scrollWheelZoom
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {points.map((p, i) => (
          <CircleMarker
            key={i}
            center={[p.lat, p.lon]}
            radius={6 + Math.log(p.count + 1) * 3}
            pathOptions={{
              color: "#2563eb",
              fillColor: "#3b82f6",
              fillOpacity: 0.7,
            }}
          >
            <Popup>
              <div className="text-sm">
                <strong>Documents:</strong> {p.count}
                <br />
                Lat: {p.lat.toFixed(2)} <br />
                Lon: {p.lon.toFixed(2)}
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}
