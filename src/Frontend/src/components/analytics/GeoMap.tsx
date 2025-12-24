import { useState, useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Button } from "@/components/ui/button";
import { SearchDocument } from "@/lib/api";

export interface GeoPoint {
  lat: number;
  lon: number;
  count: number;
  label: string;
  latest_date?: string;
  summary?: string;
  documents?: SearchDocument[];
}

interface GeoMapProps { points: GeoPoint[]; }

// ================= Helpers =================
function truncateSentence(text: string, maxLength = 260) {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  const sentences = text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
  let truncated = "";
  for (const s of sentences) {
    if ((truncated + s).length > maxLength) break;
    truncated += s + " ";
  }
  return truncated.trim() + "…";
}

function formatDate(dateStr?: string) {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  return date.toLocaleString("en-US", { year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// ================= Auto Zoom =================
function AutoFitBounds({ points }: { points: GeoPoint[] }) {
  const map = useMap();
  useEffect(() => {
    if (!points || points.length === 0) return;
    const bounds = L.latLngBounds(points.map(p => [p.lat, p.lon]));
    map.fitBounds(bounds, { padding: [60, 60], maxZoom: 5, animate: true });
  }, [points, map]);
  return null;
}

// ================= Main Map =================
export function GeoMap({ points }: GeoMapProps) {
  const [activeDoc, setActiveDoc] = useState<SearchDocument | null>(null);

  if (!points || points.length === 0) {
    return <div className="bg-muted text-muted-foreground p-4 rounded-lg">No geographic data available.</div>;
  }

  return (
    <div className="w-full h-[480px] rounded-xl overflow-hidden border shadow-sm flex gap-4">
      <MapContainer center={[points[0].lat, points[0].lon]} zoom={3} scrollWheelZoom style={{ height: "100%", width: activeDoc ? "70%" : "100%" }}>
        <AutoFitBounds points={points} />
        <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {points.map((p, i) => (
          <CircleMarker
            key={i}
            center={[p.lat, p.lon]}
            radius={7 + Math.log(p.count + 1) * 4}
            pathOptions={{ color: "#2563eb", fillColor: "#3b82f6", fillOpacity: 0.75, weight: 2 }}
            eventHandlers={{
              click: (e) => { const map = e.target._map; map.flyTo([p.lat, p.lon], Math.min(map.getZoom() + 1, 6), { animate: true, duration: 0.8 }); },
              mouseover: (e) => e.target.setStyle({ fillColor: "#1d4ed8", fillOpacity: 0.9 }),
              mouseout: (e) => e.target.setStyle({ fillColor: "#3b82f6", fillOpacity: 0.75 }),
            }}
          >
            <Popup>
              <div className="space-y-2 text-sm leading-relaxed">
                <h3 className="font-semibold text-base text-primary">📍 {p.label}</h3>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>📰 {p.count} article{p.count > 1 ? "s" : ""}</div>
                  {p.latest_date && <div>📅 Latest: {formatDate(p.latest_date)}</div>}
                </div>
                {p.summary && (
                  <div className="pt-2 border-t text-xs">
                    {truncateSentence(p.summary)}
                    {p.documents && p.documents.length > 0 && (
                      <button className="text-blue-600 underline ml-1" onClick={() => setActiveDoc(p.documents![0])}>Read more</button>
                    )}
                  </div>
                )}
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>

      {activeDoc && (
        <div className="w-1/3 bg-card border p-4 overflow-auto rounded-lg relative">
          <button onClick={() => setActiveDoc(null)} className="absolute top-2 right-2 text-sm text-muted-foreground hover:text-red-500">✕</button>
          <h3 className="font-semibold mb-2">{activeDoc.title}</h3>
          <div className="text-xs text-muted-foreground mb-2">{formatDate(activeDoc.date)} - {activeDoc.dateline}</div>
          <p className="text-sm whitespace-pre-wrap">{activeDoc.content}</p>
        </div>
      )}
    </div>
  );
}
