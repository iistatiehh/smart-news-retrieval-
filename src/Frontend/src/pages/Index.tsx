import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Header } from "@/components/layout/Header";
import { SearchBar } from "@/components/search/SearchBar";
import { FilterPanel, FilterState } from "@/components/search/FilterPanel";
import {
  Database,
  Zap,
  Globe,
  MessageSquare,
  ArrowRight,
  MapPin,
  Clock,
  Map
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, SearchResponse, SearchDocument } from "@/lib/api";
import { GeoMap } from "@/components/analytics/GeoMap";

export default function Index() {
  const navigate = useNavigate();

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showMap, setShowMap] = useState(false);

  const [filters, setFilters] = useState<FilterState>({
    dateFrom: "",
    dateTo: "",
    location: "",
    semanticSearch: true,
  });

  /* ================= Search ================= */

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    setIsLoading(true);
    setError(null);
    setSearchResult(null);
    setShowMap(false);

    try {
      let enhancedQuery = query;

      if (filters.location) {
        enhancedQuery += ` in ${filters.location}`;
      }

      if (filters.dateFrom || filters.dateTo) {
        if (filters.dateFrom && filters.dateTo) {
          enhancedQuery += ` from ${filters.dateFrom} to ${filters.dateTo}`;
        } else if (filters.dateFrom) {
          enhancedQuery += ` after ${filters.dateFrom}`;
        } else if (filters.dateTo) {
          enhancedQuery += ` before ${filters.dateTo}`;
        }
      }

      const response = await api.search(enhancedQuery);
      setSearchResult(response);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Search failed. Is the backend running?"
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleContinueInChat = () => {
    navigate(`/chat?q=${encodeURIComponent(searchQuery)}`);
  };

  /* ================= Geo Points ================= */

  const geoPoints =
    searchResult?.documents
      ?.filter(doc => doc.geo_location)
      .map(doc => ({
        lat: doc.geo_location!.lat,
        lon: doc.geo_location!.lon,
        count: 1,
      })) || [];

  /* ================= UI ================= */

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {!searchResult && !isLoading && !error && (
          <div className="max-w-3xl mx-auto text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">
              Reuters News Retrieval
            </h1>
            <p className="text-lg text-muted-foreground mb-8">
              Semantic, temporal, and geographic news exploration
            </p>
            <div className="flex justify-center gap-6 text-sm text-muted-foreground">
              <span className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                Elasticsearch
              </span>
              <span className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-primary" />
                RAG
              </span>
              <span className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-primary" />
                Geo-Search
              </span>
            </div>
          </div>
        )}

        <SearchBar onSearch={handleSearch} isLoading={isLoading} />

        {(searchResult || isLoading || error) && (
          <div className="mt-8 flex gap-6">
            {/* Filters */}
            <aside className="w-64 shrink-0 hidden lg:block">
              <FilterPanel onFilterChange={setFilters} />
            </aside>

            {/* Results */}
            <div className="flex-1 space-y-4">
              {searchResult && (
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">
                    Found{" "}
                    <span className="font-medium text-foreground">
                      {searchResult.total}
                    </span>{" "}
                    documents for{" "}
                    <span className="font-medium text-foreground">
                      {searchQuery}
                    </span>
                  </p>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowMap(prev => !prev)}
                      disabled={geoPoints.length === 0}
                    >
                      <Map className="w-4 h-4 mr-2" />
                      {showMap ? "Hide Map" : "Show Map"}
                    </Button>

                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleContinueInChat}
                    >
                      <MessageSquare className="w-4 h-4 mr-2" />
                      Ask AI
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              )}

              {/* 🗺️ Map */}
              {showMap && geoPoints.length > 0 && (
                <GeoMap points={geoPoints} />
              )}

              {/* 📄 Documents */}
              {searchResult?.documents.map(doc => (
                <DocumentCard key={doc.id} document={doc} />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

/* ================= Document Card ================= */

interface DocumentCardProps {
  document: SearchDocument;
}

function DocumentCard({ document }: DocumentCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="bg-card border rounded-lg p-6">
      <h3 className="font-semibold mb-1">{document.title}</h3>

      <div className="flex gap-3 text-xs text-muted-foreground mb-3">
        {document.date && (
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {document.date}
          </span>
        )}
        {document.dateline && (
          <span className="flex items-center gap-1">
            <MapPin className="w-3 h-3" />
            {document.dateline}
          </span>
        )}
      </div>

      <p className={`text-sm mb-3 ${expanded ? "" : "line-clamp-3"}`}>
        {document.content}
      </p>

      {document.content.length > 300 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-primary"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </article>
  );
}
