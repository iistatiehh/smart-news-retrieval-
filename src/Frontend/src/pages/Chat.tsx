import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Header } from "@/components/layout/Header";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { RecentSessions } from "@/components/chat/RecentSessions"; // ✅ CHANGED
import { api } from "@/lib/api";
import { AlertCircle, Loader2 } from "lucide-react";

function generateSessionId() {
  return `sess_${Math.random().toString(36).substring(2, 10)}`;
}

function getOrCreateSessionId(): string {
  const stored = localStorage.getItem('chat_session_id');
  if (stored) {
    return stored;
  }
  const newId = generateSessionId();
  localStorage.setItem('chat_session_id', newId);
  return newId;
}

// ✅ ADD: Save session to history
function saveSessionToHistory(sessionId: string) {
  const messages = localStorage.getItem(`chat_messages_${sessionId}`);
  if (!messages) return;

  try {
    const parsed = JSON.parse(messages);
    if (parsed.length <= 1) return; // Don't save if only welcome message

    // Get first user message
    const firstUserMessage = parsed.find((m: any) => m.role === "user");
    if (!firstUserMessage) return;

    const sessionData = {
      id: sessionId,
      firstMessage: firstUserMessage.content.substring(0, 100), // Truncate to 100 chars
      lastActivity: new Date().toISOString(),
      messageCount: parsed.filter((m: any) => m.role === "user").length,
    };

    // Load existing sessions
    const existingSessions = localStorage.getItem('chat_sessions_list');
    const sessions = existingSessions ? JSON.parse(existingSessions) : [];

    // Remove if already exists (update)
    const filtered = sessions.filter((s: any) => s.id !== sessionId);

    // Add to beginning
    filtered.unshift(sessionData);

    // Keep only 20 most recent
    const limited = filtered.slice(0, 20);

    localStorage.setItem('chat_sessions_list', JSON.stringify(limited));
  } catch (err) {
    console.error('Failed to save session to history:', err);
  }
}

export default function Chat() {
  const [searchParams] = useSearchParams();
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [backendStatus, setBackendStatus] = useState<"checking" | "connected" | "disconnected">("checking");
  
  const initialQuery = searchParams.get("q") || undefined;

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const health = await api.healthCheck();
      setBackendStatus(health.status === "healthy" ? "connected" : "disconnected");
    } catch {
      setBackendStatus("disconnected");
    }
  };

  const handleNewSession = () => {
    // ✅ CHANGED: Save current session before creating new one
    saveSessionToHistory(sessionId);
    
    const newId = generateSessionId();
    setSessionId(newId);
    localStorage.setItem('chat_session_id', newId);
    localStorage.removeItem(`chat_messages_${sessionId}`); // Clear old messages from this variable
  };

  const handleClearSession = () => {
    localStorage.removeItem(`chat_messages_${sessionId}`);
    api.clearSession(sessionId);
  };

  // ✅ ADD: Load a previous session
  const handleLoadSession = (oldSessionId: string) => {
    // Save current session first
    saveSessionToHistory(sessionId);
    
    // Switch to old session
    setSessionId(oldSessionId);
    localStorage.setItem('chat_session_id', oldSessionId);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />

      {/* Backend Status Banner */}
      {backendStatus !== "connected" && (
        <div className={`px-4 py-2 text-sm flex items-center justify-center gap-2 ${
          backendStatus === "checking" 
            ? "bg-muted text-muted-foreground" 
            : "bg-destructive/10 text-destructive"
        }`}>
          {backendStatus === "checking" ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Connecting to backend...
            </>
          ) : (
            <>
              <AlertCircle className="w-4 h-4" />
              Backend unavailable. Make sure the API server is running on localhost:8000
            </>
          )}
        </div>
      )}

      <div className="flex-1 container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-8rem)]">
          {/* Recent Sessions - Left Sidebar */}
          <aside className="hidden lg:block">
            <RecentSessions 
              currentSessionId={sessionId} 
              onLoadSession={handleLoadSession}
            />
          </aside>

          {/* Main Chat Area */}
          <div className="lg:col-span-3 bg-card border border-border rounded-xl overflow-hidden">
            <ChatInterface
              sessionId={sessionId}
              onNewSession={handleNewSession}
              onClearSession={handleClearSession}
              initialQuery={initialQuery}
            />
          </div>
        </div>
      </div>
    </div>
  );
}