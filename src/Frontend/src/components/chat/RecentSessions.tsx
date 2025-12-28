import { MessageSquare, Clock, Trash2 } from "lucide-react";
import { useState, useEffect } from "react";

interface SavedSession {
  id: string;
  firstMessage: string;
  lastActivity: string;
  messageCount: number;
}

interface RecentSessionsProps {
  currentSessionId: string;
  onLoadSession: (sessionId: string) => void;
}

export function RecentSessions({ currentSessionId, onLoadSession }: RecentSessionsProps) {
  const [sessions, setSessions] = useState<SavedSession[]>([]);

  useEffect(() => {
    loadSessions();
  }, [currentSessionId]);

  const loadSessions = () => {
    const savedSessions = localStorage.getItem('chat_sessions_list');
    if (savedSessions) {
      try {
        const parsed: SavedSession[] = JSON.parse(savedSessions);
        // Filter out current session and sort by last activity
        const filtered = parsed
          .filter(s => s.id !== currentSessionId)
          .sort((a, b) => new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime())
          .slice(0, 10); // Keep only 10 most recent
        setSessions(filtered);
      } catch (err) {
        console.error('Failed to load sessions:', err);
      }
    }
  };

  const deleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    
    // Remove from list
    const savedSessions = localStorage.getItem('chat_sessions_list');
    if (savedSessions) {
      const parsed: SavedSession[] = JSON.parse(savedSessions);
      const filtered = parsed.filter(s => s.id !== sessionId);
      localStorage.setItem('chat_sessions_list', JSON.stringify(filtered));
    }
    
    // Remove session messages
    localStorage.removeItem(`chat_messages_${sessionId}`);
    
    // Refresh list
    loadSessions();
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "just now";
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    return date.toLocaleDateString();
  };

  if (sessions.length === 0) {
    return (
      <div className="bg-card border border-border rounded-xl overflow-hidden h-full">
        <div className="p-4 border-b border-border">
          <h3 className="font-semibold text-sm">Recent Sessions</h3>
        </div>
        <div className="p-6 text-center text-sm text-muted-foreground">
          <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p>No previous sessions</p>
          <p className="text-xs mt-1">Start a new chat to see history here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h3 className="font-semibold text-sm">Recent Sessions</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {sessions.map((session) => (
          <div
            key={session.id}
            className="relative group"
          >
            {/* ✅ MAIN SESSION BUTTON */}
            <button
              onClick={() => onLoadSession(session.id)}
              className="w-full text-left p-3 rounded-lg hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-start gap-3">
                <MessageSquare className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
                <div className="min-w-0 flex-1 pr-8">
                  <p className="text-sm font-medium text-foreground line-clamp-2 leading-snug">
                    {session.firstMessage}
                  </p>
                  <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                    <span>{session.messageCount} message{session.messageCount !== 1 ? 's' : ''}</span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatTime(session.lastActivity)}
                    </span>
                  </div>
                </div>
              </div>
            </button>
            
            {/* ✅ DELETE BUTTON - SEPARATE, POSITIONED ABSOLUTELY */}
            <button
              onClick={(e) => deleteSession(session.id, e)}
              className="absolute top-2 right-2 p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all z-10"
              title="Delete session"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
