import { useState, useEffect } from "react"; // ✅ ADD useEffect here
import { useNavigate, useSearchParams } from "react-router-dom"; // ✅ ADD useSearchParams
import { Header } from "@/components/layout/Header";
import { SearchBar } from "@/components/search/SearchBar";
import { FilterPanel, FilterState } from "@/components/search/FilterPanel";
import { GeoMap, GeoPoint } from "@/components/analytics/GeoMap";
import { Database, Zap, Globe, MessageSquare, ArrowRight, MapPin, Clock, FileText, Building, Users, Map as MapIcon } from "lucide-react"; // ✅ ADD Map icon
import { Button } from "@/components/ui/button";
import { api, SearchResponse, SearchDocument } from "@/lib/api";

export default function Index() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams(); // ✅ ADD THIS
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showMap, setShowMap] = useState(false); // ✅ ADD THIS
  const [filters, setFilters] = useState<FilterState>({
    dateFrom: "",
    dateTo: "",
    location: "",
    semanticSearch: true,
  });

  // ✅ ADD THIS: Transform search results into GeoPoints
  const getGeoPoints = (): GeoPoint[] => {
    if (!searchResult?.documents) return [];
    
    const locationMap = new Map<string, GeoPoint>();
    
    searchResult.documents.forEach(doc => {
      if (doc.geo_location?.lat && doc.geo_location?.lon) {
        const key = `${doc.geo_location.lat},${doc.geo_location.lon}`;
        
        if (locationMap.has(key)) {
          const existing = locationMap.get(key)!;
          existing.count += 1;
          existing.documents?.push(doc);
          // Update latest date if newer
          if (doc.date && (!existing.latest_date || doc.date > existing.latest_date)) {
            existing.latest_date = doc.date;
            existing.summary = doc.content.substring(0, 200);
          }
        } else {
          locationMap.set(key, {
            lat: doc.geo_location.lat,
            lon: doc.geo_location.lon,
            count: 1,
            label: doc.dateline || doc.places[0] || "Unknown Location",
            latest_date: doc.date,
            summary: doc.content.substring(0, 200),
            documents: [doc]
          });
        }
      }
    });
    
    return Array.from(locationMap.values());
  };
  useEffect(() => {
    const queryFromUrl = searchParams.get('q');
    if (queryFromUrl && !searchResult) {
      handleSearch(queryFromUrl);
    }
  }, [searchParams]); // Only run when URL changes

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    setIsLoading(true);
    setError(null);
    setSearchResult(null);
    setShowMap(false); // ✅ ADD THIS: Reset map visibility on new search
    setSearchParams({ q: query });

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
      setError(err instanceof Error ? err.message : "Search failed. Is the backend running?");
    } finally {
      setIsLoading(false);
    }
  };

  const handleContinueInChat = () => {
    navigate(`/chat?q=${encodeURIComponent(searchQuery)}`);
  };

  const formatScore = (score: number) => {
    return Math.min(100, Math.round(score * 10)).toString();
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section - Keep as is */}
        {!searchResult && !isLoading && !error && (
          <div className="max-w-3xl mx-auto text-center mb-12 animate-fade-in">
            <h1 className="text-4xl font-bold text-foreground mb-4">
              Reuters News Retrieval
            </h1>
            <p className="text-lg text-muted-foreground mb-8">
              Search 21,000+ Reuters articles here or use our chatting model with semantic understanding, 
              temporal reasoning, and geographic context.
            </p>
            <div className="flex justify-center gap-6 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                <span>Elasticsearch Backend</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-primary" />
                <span>RAG-Powered</span>
              </div>
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-primary" />
                <span>Spatiotemporal Search</span>
              </div>
            </div>
          </div>
        )}

        <SearchBar onSearch={handleSearch} isLoading={isLoading} />

        {(searchResult || isLoading || error) && (
          <div className="mt-8 flex gap-6">
            <aside className="w-64 shrink-0 hidden lg:block">
              <FilterPanel onFilterChange={setFilters} />
            </aside>

            <div className="flex-1">
              {isLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="bg-card border border-border rounded-lg p-6 animate-pulse">
                      <div className="h-5 bg-muted rounded w-3/4 mb-3" />
                      <div className="h-3 bg-muted rounded w-1/4 mb-4" />
                      <div className="space-y-2">
                        <div className="h-4 bg-muted rounded w-full" />
                        <div className="h-4 bg-muted rounded w-5/6" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : error ? (
                <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-6 text-center">
                  <p className="text-destructive font-medium mb-2">Search Error</p>
                  <p className="text-sm text-muted-foreground">{error}</p>
                  <p className="text-xs text-muted-foreground mt-4">
                    Make sure the FastAPI backend is running on <code className="bg-muted px-1 py-0.5 rounded">localhost:8000</code>
                  </p>
                </div>
              ) : searchResult ? (
                <div className="space-y-4 animate-fade-in">
                  {/* ✅ MODIFIED: Search Info with Map Toggle */}
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-muted-foreground">
                      Found <span className="font-medium text-foreground">{searchResult.total}</span> documents for "
                      <span className="font-medium text-foreground">{searchQuery}</span>"
                    </p>
                    <div className="flex gap-2">
                      {/* ✅ ADD THIS: Map Toggle Button */}
                      {getGeoPoints().length > 0 && (
                        <Button 
                          variant="outline" 
                          size="sm" 
                          onClick={() => setShowMap(!showMap)}
                        >
                          <MapIcon className="w-4 h-4 mr-2" />
                          {showMap ? "Hide Map" : "Show Map"}
                        </Button>
                      )}
                      <Button variant="outline" size="sm" onClick={handleContinueInChat}>
                        <MessageSquare className="w-4 h-4 mr-2" />
                        Ask AI about results
                        <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  </div>

                  {/* ✅ ADD THIS: Conditional Map Display */}
                  {showMap && getGeoPoints().length > 0 && (
                    <div className="mb-6">
                      <GeoMap points={getGeoPoints()} />
                    </div>
                  )}

                  {/* Document Results - Keep as is */}
                  {searchResult.documents.map((doc) => (
                    <DocumentCard key={doc.id} document={doc} formatScore={formatScore} />
                  ))}

                  {/* Continue to Chat CTA - Keep as is */}
                  <div className="bg-primary/5 border border-primary/20 rounded-lg p-6 text-center">
                    <MessageSquare className="w-8 h-8 text-primary mx-auto mb-3" />
                    <h3 className="font-semibold mb-2">Want to explore further?</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Ask the AI assistant questions about these documents
                    </p>
                    <Button onClick={handleContinueInChat}>
                      Open in Chat
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

interface DocumentCardProps {
  document: SearchDocument;
  formatScore: (score: number) => string;
}

function DocumentCard({ document, formatScore }: DocumentCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="bg-card border border-border rounded-lg p-6 hover:border-primary/30 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-foreground leading-snug line-clamp-2 mb-1">
            {document.title}
          </h3>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
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
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="text-xs text-muted-foreground">Relevance</span>
          <div className="flex items-center gap-2">
            <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${document.relevanceScore * 100}%` }} 
              />
            </div>
            <span className="text-xs font-medium text-foreground w-8">{Math.round(document.relevanceScore * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <p className={`text-sm text-muted-foreground leading-relaxed mb-4 ${expanded ? '' : 'line-clamp-3'}`}>
        {document.content}
      </p>
      {document.content.length > 300 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-primary hover:underline mb-4"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}

      {/* Metadata Tags */}
      <div className="flex flex-wrap gap-2">
        {document.places.slice(0, 3).map((place, i) => (
          <span key={i} className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded-full">
            <MapPin className="w-3 h-3" />
            {place}
          </span>
        ))}
        {document.temporalExpressions.slice(0, 2).map((expr, i) => (
          <span key={i} className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-amber-500/10 text-amber-600 dark:text-amber-400 rounded-full">
            <Clock className="w-3 h-3" />
            {expr}
          </span>
        ))}
        {document.topics.slice(0, 2).map((topic, i) => (
          <span key={i} className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-full">
            <FileText className="w-3 h-3" />
            {topic}
          </span>
        ))}
        {document.orgs.slice(0, 2).map((org, i) => (
          <span key={i} className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-purple-500/10 text-purple-600 dark:text-purple-400 rounded-full">
            <Building className="w-3 h-3" />
            {org}
          </span>
        ))}
        {document.people.slice(0, 2).map((person, i) => (
          <span key={i} className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-rose-500/10 text-rose-600 dark:text-rose-400 rounded-full">
            <Users className="w-3 h-3" />
            {person}
          </span>
        ))}
      </div>
    </article>
  );
}
