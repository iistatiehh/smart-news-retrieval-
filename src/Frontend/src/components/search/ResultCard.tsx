import { MapPin, Clock, Lightbulb, FileText, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface SearchResult {
  id: string;
  title: string;
  date: string;
  content: string;
  score: number;
  places: string[];
  temporalExpressions: string[];
  georeferences?: string[];
  dateline?: string;
}

interface ResultCardProps {
  result: SearchResult;
  index: number;
  onExplain: (result: SearchResult) => void;
  onSummarize: (result: SearchResult) => void;
  onAsk: (result: SearchResult) => void;
}

export function ResultCard({ result, index, onExplain, onSummarize, onAsk }: ResultCardProps) {
  console.log('ResultCard render:', {
    title: result.title,
    score: result.score,
    places: result.places,
    georeferences: result.georeferences,
    dateline: result.dateline
  });

  const getRelevanceColor = (score: number) => {
    if (score >= 150) return "bg-green-500";
    if (score >= 100) return "bg-yellow-500";
    return "bg-red-500";
  };

  const normalizeScore = (score: number) => {
    const maxScore = 200;
    return Math.min((score / maxScore) * 100, 100);
  };

  const locations = result.georeferences && result.georeferences.length > 0 
    ? result.georeferences 
    : result.places;

  const dateInfo = result.dateline || result.date.split('T')[0];

  return (
    <article 
      className="border rounded-lg p-4 bg-white shadow-sm"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900 leading-snug mb-1">
            {result.title}
          </h3>
          <time className="text-xs text-gray-500">{result.date}</time>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className="text-xs text-gray-500">Score: {result.score.toFixed(1)}</span>
          <div className="w-16 h-2 bg-gray-200 rounded overflow-hidden">
            <div
              className={`h-full ${getRelevanceColor(result.score)}`}
              style={{ width: `${normalizeScore(result.score)}%` }}
            />
          </div>
        </div>
      </div>

      <p className="text-sm text-gray-600 leading-relaxed mb-4">
        {result.content.substring(0, 200)}...
      </p>

      {/* Debug section */}
      <div className="mb-2 p-2 bg-gray-100 text-xs">
        <div>Places: {JSON.stringify(result.places)}</div>
        <div>Georeferences: {JSON.stringify(result.georeferences)}</div>
        <div>Dateline: {result.dateline}</div>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {locations && locations.map((place, i) => (
          <span key={i} className="flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
            <MapPin className="w-3 h-3" />
            {place}
          </span>
        ))}
        {dateInfo && (
          <span className="flex items-center gap-1 px-2 py-1 bg-amber-100 text-amber-700 rounded text-xs">
            <Clock className="w-3 h-3" />
            {dateInfo}
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-2 pt-3 border-t">
        <Button variant="ghost" size="sm" onClick={() => onExplain(result)}>
          <Lightbulb className="w-3.5 h-3.5 mr-1.5" />
          Explain
        </Button>
        <Button variant="ghost" size="sm" onClick={() => onSummarize(result)}>
          <FileText className="w-3.5 h-3.5 mr-1.5" />
          Summarize
        </Button>
        <Button variant="ghost" size="sm" onClick={() => onAsk(result)}>
          <MessageCircle className="w-3.5 h-3.5 mr-1.5" />
          Ask
        </Button>
      </div>
    </article>
  );
}