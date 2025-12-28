import { useState, useRef, useEffect } from "react";
import { Send, RotateCcw, Plus, FileText, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, ChatResponse } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  queryRewritten?: string | null;
  documentsUsed?: string[];
  isError?: boolean;
}

interface ChatInterfaceProps {
  sessionId: string;
  onNewSession: () => void;
  onClearSession: () => void;
  initialQuery?: string;
}

export function ChatInterface({ sessionId, onNewSession, onClearSession, initialQuery }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const initialQueryProcessed = useRef(false);

  // ✅ ADD: Load messages from localStorage on mount or when sessionId changes
  useEffect(() => {
    const savedMessages = localStorage.getItem(`chat_messages_${sessionId}`);
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        setMessages(parsed);
      } catch (err) {
        console.error("Failed to parse saved messages:", err);
        // Fallback to welcome message
        setMessages([{
          id: "welcome",
          role: "assistant",
          content: "Hello! I'm your research assistant for the Reuters news archive. I can help you explore news articles, answer questions about specific events, and find connections across documents. What would you like to know?",
        }]);
      }
    } else {
      // No saved messages, show welcome
      setMessages([{
        id: "welcome",
        role: "assistant",
        content: "Hello! I'm your research assistant for the Reuters news archive. I can help you explore news articles, answer questions about specific events, and find connections across documents. What would you like to know?",
      }]);
    }
  }, [sessionId]);

  // ✅ ADD: Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(`chat_messages_${sessionId}`, JSON.stringify(messages));
    }
  }, [messages, sessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle initial query from URL
  useEffect(() => {
    if (initialQuery && !initialQueryProcessed.current) {
      initialQueryProcessed.current = true;
      handleSendMessage(initialQuery);
    }
  }, [initialQuery]);

  const handleSendMessage = async (messageContent: string) => {
    if (!messageContent.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: messageContent,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response: ChatResponse = await api.chat(messageContent, sessionId, true);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        queryRewritten: response.query_rewritten,
        documentsUsed: response.documents_used,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: error instanceof Error ? error.message : "Failed to get response. Please try again.",
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(input);
  };

  const handleClearSession = async () => {
    try {
      await api.clearSession(sessionId);
      
      // ✅ MODIFIED: Clear localStorage when clearing session
      localStorage.removeItem(`chat_messages_${sessionId}`);
      
      setMessages([
        {
          id: "welcome",
          role: "assistant",
          content: "Conversation cleared. How can I help you?",
        },
      ]);
      onClearSession();
    } catch (error) {
      console.error("Failed to clear session:", error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border bg-card">
        <div>
          <h2 className="font-semibold">Conversational Search</h2>
          <p className="text-xs text-muted-foreground font-mono">Session: {sessionId}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleClearSession} disabled={isLoading}>
            <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
            Clear
          </Button>
          <Button variant="outline" size="sm" onClick={onNewSession} disabled={isLoading}>
            <Plus className="w-3.5 h-3.5 mr-1.5" />
            New Session
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
          >
            <div className={`max-w-[80%] ${message.role === "user" ? "order-2" : ""}`}>
              <div
                className={`${message.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant"} ${
                  message.isError ? "border border-destructive/50 bg-destructive/10" : ""
                }`}
              >
                {message.isError && (
                  <div className="flex items-center gap-2 text-destructive mb-2">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-xs font-medium">Error</span>
                  </div>
                )}
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({ children }) => (
                        <div className="overflow-x-auto my-4 rounded-lg border border-border">
                          <table className="w-full text-sm border-collapse">{children}</table>
                        </div>
                      ),
                      thead: ({ children }) => (
                        <thead className="bg-muted/50">{children}</thead>
                      ),
                      th: ({ children }) => (
                        <th className="px-3 py-2 text-left font-semibold text-foreground border-b border-border">{children}</th>
                      ),
                      td: ({ children }) => (
                        <td className="px-3 py-2 border-b border-border/50 text-foreground/90">{children}</td>
                      ),
                      tr: ({ children }) => (
                        <tr className="hover:bg-muted/30 transition-colors">{children}</tr>
                      ),
                      p: ({ children }) => (
                        <p className="text-sm leading-relaxed mb-3 last:mb-0">{children}</p>
                      ),
                      strong: ({ children }) => (
                        <strong className="font-semibold text-foreground">{children}</strong>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-disc list-inside space-y-1 my-3">{children}</ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal list-inside space-y-1 my-3">{children}</ol>
                      ),
                      li: ({ children }) => (
                        <li className="text-sm text-foreground/90">{children}</li>
                      ),
                      h1: ({ children }) => (
                        <h1 className="text-lg font-bold mb-3 mt-4 text-foreground">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-base font-bold mb-2 mt-4 text-foreground">{children}</h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-sm font-bold mb-2 mt-3 text-foreground">{children}</h3>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-2 border-primary/50 pl-4 my-3 italic text-muted-foreground">{children}</blockquote>
                      ),
                      code: ({ children, className }) => {
                        const isInline = !className;
                        return isInline ? (
                          <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
                        ) : (
                          <code className="block bg-muted p-3 rounded-lg text-xs font-mono overflow-x-auto my-3">{children}</code>
                        );
                      },
                      hr: () => <hr className="my-4 border-border" />,
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              </div>

              {/* Query Rewritten Indicator */}
              {message.queryRewritten && (
                <div className="mt-2 px-3 py-1.5 bg-muted/50 rounded-lg text-xs text-muted-foreground">
                  <span className="font-medium">Query interpreted as:</span> {message.queryRewritten}
                </div>
              )}

              {/* Documents Used */}
              {message.documentsUsed && message.documentsUsed.length > 0 && (
                <div className="mt-3 space-y-2">
                  <p className="text-xs font-medium text-muted-foreground px-1">
                    Sources ({message.documentsUsed.length}):
                  </p>
                  {message.documentsUsed.map((doc, i) => (
                    <div
                      key={i}
                      className="bg-card border border-border rounded-lg p-3 text-sm"
                    >
                      <div className="flex items-start gap-2">
                        <FileText className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                        <p className="font-medium text-foreground line-clamp-2">{doc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start animate-fade-in">
            <div className="chat-bubble-assistant">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-primary" />
                <span className="text-sm text-muted-foreground">Searching documents...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-border bg-card">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about the news archive…"
            className="flex-1 px-4 py-3 text-sm bg-background border border-border rounded-xl focus:border-primary focus:ring-1 focus:ring-primary/20 transition-colors"
            disabled={isLoading}
          />
          <Button type="submit" disabled={!input.trim() || isLoading} className="px-4">
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </div>
      </form>
    </div>
  );
}