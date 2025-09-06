"use client";

import React, { useEffect, useRef, useState } from "react";
import { Send, Trash2, Bot, User, ChevronDown, ChevronUp, Copy, ArrowDown, ArrowUp, FileText, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { API_CONFIG } from "@/lib/api-config";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";

const TypingAnswer = ({ text, onDone }: { text: string; onDone: (finalText: string) => void }) => {
  const [displayed, setDisplayed] = useState("");
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; }, [onDone]);
  useEffect(() => {
    let i = 0;
    const chunk = 2;
    const pace = 10;
    const interval = setInterval(() => {
      const next = text.slice(i, i + chunk);
      setDisplayed((prev) => prev + next);
      i += chunk;
      if (i >= text.length) {
        clearInterval(interval);
        setDisplayed(text);
        onDoneRef.current(text);
      }
    }, pace);
    return () => clearInterval(interval);
  }, [text]);

  return (
    <div className="whitespace-pre-wrap">
      <ReactMarkdown 
        remarkPlugins={[remarkGfm, remarkMath]} 
        rehypePlugins={[rehypeKatex]}
        components={{
          table: ({ node, ...props }) => (
            <table {...props} className="border-collapse border border-border my-4 w-full rounded-md overflow-hidden" />
          ),
          th: ({ node, ...props }) => (
            <th {...props} className="border border-border px-3 py-2 bg-secondary text-secondary-foreground font-semibold text-left" />
          ),
          td: ({ node, ...props }) => (
            <td {...props} className="border border-border px-3 py-2" />
          ),
        }}
      >
        {displayed}
      </ReactMarkdown>
      {displayed.length < text.length && <span className="animate-pulse"></span>}
    </div>
  );
};

interface Citation {
  chunk_id: string;
  page?: number;
  text: string;
  image?: string;
  pdf?: string;
}

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
  sources?: Citation[];
  showCitations?: boolean;
  isTyping?: boolean;
  typingText?: string;
  showCitationsFadeIn?: boolean;
}

interface ChatUIProps {
  pdfName?: string;
}

export function ChatUI({ pdfName }: ChatUIProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [showScrollToTop, setShowScrollToTop] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [summaryLength, setSummaryLength] = useState<'short' | 'medium' | 'detailed'>("short");

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const scrollToTop = () => {
    scrollAreaRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
    const isAtTop = scrollTop < 10;
    
    setShowScrollToBottom(!isAtBottom && scrollHeight > clientHeight);
    setShowScrollToTop(!isAtTop && scrollHeight > clientHeight);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const historyPayload = messages
      .slice(-10)
      .map((m) => ({
        role: m.sender === "user" ? "user" : "assistant",
        content: m.content,
      }));

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      sender: "user",
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const askUrl = API_CONFIG.getUrl("ask");
      const response = await fetch(askUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.content,
          pdfName: pdfName,
          pastMessages: historyPayload,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const full = data.answer || "I'm sorry, I couldn't process your question.";
        const aiId = (Date.now() + 1).toString();
        // Insert message in typing mode; content fills in when animation completes
        setMessages(prev => [
          ...prev,
          {
            id: aiId,
            content: "",
            sender: "ai",
            timestamp: new Date(),
            sources: Array.isArray(data.sources) ? data.sources : [],
            showCitations: false,
            isTyping: true,
            typingText: full,
          },
        ]);
      } else {
        let errorMessage = "Failed to get response. Please try again.";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          const errorText = await response.text();
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorMessage;
          } catch {
            errorMessage = errorText || errorMessage;
          }
        }
        const errorAiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: `Error: ${errorMessage}`,
          sender: "ai",
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorAiMessage]);
        toast.error(errorMessage);
      }
    } catch (error: any) {
      const errorMessage = error?.message || "Failed to communicate with the API";
      const errorAiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Error: ${errorMessage}`,
        sender: "ai",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorAiMessage]);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSummary(null);
  };

  const handleSummarize = async () => {
    if (!pdfName || isSummarizing) return;

    setIsSummarizing(true);
    try {
      const summarizeUrl = API_CONFIG.getUrl("summarize");
      const response = await fetch(summarizeUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pdfName: pdfName,
          summary_length: summaryLength,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSummary(data.summary || "No summary available.");
        toast.success("Document summarized successfully!");
      } else {
        let errorMessage = "Failed to summarize";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
        } catch {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        }
        toast.error(`Summarization failed: ${errorMessage}`);
      }
    } catch (error: any) {
      toast.error(`Error: ${error.message || "Failed to summarize document"}`);
    } finally {
      setIsSummarizing(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success("Copied to clipboard!");
    }).catch(() => {
      toast.error("Failed to copy to clipboard");
    });
  };

  const toggleCitations = (messageId: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, showCitations: !msg.showCitations }
        : msg
    ));
  };

  const markdownToPlain = (md: string): string => {
    let out = md;
    
    out = out.replace(/^#{1,6}\s+/gm, '');
    
    out = out.replace(/\*\*(.*?)\*\*/g, '$1');
    out = out.replace(/\*(.*?)\*/g, '$1');
    
    out = out.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    
    out = out.replace(/```[\s\S]*?```/g, '');
    out = out.replace(/`([^`]+)`/g, '$1');
    
    out = out.replace(/\n{3,}/g, '\n\n');
    return out.trim();
  };

  const ThinkingDots = () => (
    <div className="rounded-lg p-3 bg-muted">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-foreground/70 animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-foreground/60 animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-foreground/50 animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
        <span className="text-sm text-muted-foreground">Thinking…</span>
      </div>
    </div>
  );

  return (
    <Card className="flex flex-col h-[600px] w-full max-w-4xl mx-auto relative">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <Bot className="h-6 w-6" />
          <h2 className="text-lg font-semibold">
            {pdfName ? `Chat with ${pdfName}` : "PDF Q&A Assistant"}
          </h2>
          {pdfName && (
            <Badge variant="secondary" className="ml-1 max-w-[160px] truncate" title={pdfName}>
              {pdfName}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          {pdfName && (
            <TooltipProvider>
              <div className="flex items-center gap-2 bg-muted/50 border rounded-lg px-2 py-1">
                <span className="text-xs text-muted-foreground">Length</span>
                <Select
                  value={summaryLength}
                  onValueChange={(v) => setSummaryLength(v as 'short' | 'medium' | 'detailed')}
                  disabled={isSummarizing}
                >
                  <SelectTrigger className="h-8 w-[140px]" aria-label="Choose summary length">
                    <SelectValue placeholder="Summary length" />
                  </SelectTrigger>
                  <SelectContent align="end">
                    <SelectItem value="short">Short</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="detailed">Detailed</SelectItem>
                  </SelectContent>
                </Select>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={handleSummarize}
                      disabled={isSummarizing}
                      title="Summarize document"
                      aria-label="Summarize document"
                    >
                      {isSummarizing ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Summarizing...
                        </>
                      ) : (
                        <>
                          <FileText className="h-4 w-4 mr-2" />
                          Summarize
                        </>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">Generate a {summaryLength} summary</TooltipContent>
                </Tooltip>
              </div>
            </TooltipProvider>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="sm" onClick={clearChat} title="Clear chat" aria-label="Clear chat">
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Chat
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Remove all messages</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      <ScrollArea className="flex-1 p-4" onScrollCapture={handleScroll}>
        <div ref={scrollAreaRef} className="space-y-4">
          {summary && (
            <div className="mb-4 rounded-md border bg-background">
              <div className="px-3 py-2 border-b text-sm font-medium flex items-center justify-between">
                <span className="inline-flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Document Summary ({summaryLength})
                </span>
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="ghost" onClick={() => copyToClipboard(summary || '')}>
                    <Copy className="h-4 w-4 mr-1" /> Copy
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setSummary(null)}>Clear</Button>
                </div>
              </div>
              <div className="p-3 text-sm">
                <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {summary}
                </ReactMarkdown>
              </div>
            </div>
          )}
          
          {messages.length === 0 ? (
            <div className="text-center py-8">
              <Bot className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-muted-foreground">
                Ask me anything about your PDF document!
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`flex items-start space-x-3 max-w-[80%] ${
                      message.sender === "user" ? "flex-row-reverse space-x-reverse" : ""
                    }`}
                  >
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                      {message.sender === "user" ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </div>
                    <div
                      className={`rounded-lg p-3 relative ${
                        message.sender === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted"
                      }`}
                    >
                      {message.sender === "ai" ? (
                        <div className="text-sm space-y-2">
                          {/* Copy appears only after typing complete and content exists */}
                          {(!message.isTyping && message.content && message.content.trim().length > 0) && (
                            <div className="flex justify-end">
                              <Button
                                variant="ghost"
                                size="icon"
                                title="Copy answer"
                                onClick={() => copyToClipboard(markdownToPlain(message.content))}
                              >
                                <Copy className="h-4 w-4" />
                              </Button>
                            </div>
                          )}
                          {message.isTyping && message.typingText ? (
                            <TypingAnswer
                              text={message.typingText}
                              onDone={(finalText) =>
                                setMessages((prev) =>
                                  prev.map((m) =>
                                    m.id === message.id
                                      ? { ...m, content: finalText, isTyping: false, typingText: undefined, showCitationsFadeIn: true }
                                      : m
                                  )
                                )
                              }
                            />
                          ) : (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                            components={{
                              a: ({ node, ...props }) => (
                                <a
                                  {...props}
                                  className="underline"
                                  target="_blank"
                                  rel="noreferrer"
                                />
                              ),
                              code: ({ node, inline, ...props }) => (
                                inline ? (
                                  <code {...props} className="bg-muted px-1 py-0.5 rounded text-sm" />
                                ) : (
                                  <code {...props} className="block bg-muted p-2 rounded text-sm overflow-x-auto" />
                                )
                              ),
                              table: ({ node, ...props }) => (
                                <table {...props} className="border-collapse border border-border my-4 w-full rounded-md overflow-hidden" />
                              ),
                              th: ({ node, ...props }) => (
                                <th {...props} className="border border-border px-3 py-2 bg-secondary text-secondary-foreground font-semibold text-left" />
                              ),
                              td: ({ node, ...props }) => (
                                <td {...props} className="border border-border px-3 py-2" />
                              ),
                              li: ({ node, ordered, ...props }) => (
                                <li {...props} className="ml-4 list-disc" />
                              ),
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                          )}
                          {Array.isArray(message.sources) && message.sources.length > 0 && (
                            <div className="mt-3 border-t border-muted-foreground/20 pt-2">
                              <div className={`flex items-center justify-between mb-1 transition-all duration-700 ${
                                message.showCitationsFadeIn ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
                              }`}>
                                <div className="text-xs font-medium">Citations</div>
                                <button
                                  type="button"
                                  aria-label={message.showCitations ? "Hide citations" : "Show citations"}
                                  title={message.showCitations ? "Hide citations" : "Show citations"}
                                  onClick={() => toggleCitations(message.id)}
                                  className="bg-background/80 hover:bg-background shadow rounded-full w-7 h-7 md:w-8 md:h-8 flex items-center justify-center border border-muted-foreground/20"
                                >
                                  {message.showCitations ? (
                                    <ChevronUp className="h-3 w-3" />
                                  ) : (
                                    <ChevronDown className="h-3 w-3" />
                                  )}
                                </button>
                              </div>
                              {message.showCitations && (
                                <ul className="space-y-1">
                                  {message.sources.slice(0, Math.min(5, message.sources.length)).map((src) => {
                                    const href = src.pdf
                                      ? `${API_CONFIG.baseUrl}/media/${src.pdf}${src.page ? `#page=${src.page}` : ''}`
                                      : undefined;
                                    return (
                                      <li
                                        key={src.chunk_id}
                                        className={`text-xs text-muted-foreground transition-all duration-700 delay-200 ${
                                          message.showCitationsFadeIn ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
                                        }`}
                                      >
                                        <span className="mr-1">📖</span>
                                        {src.page && href ? (
                                          <>
                                            <a
                                              href={href}
                                              target="_blank"
                                              rel="noreferrer"
                                              className="underline underline-offset-2"
                                            >
                                              {`Page ${src.page}`} ↗
                                            </a>
                                            <span className="ml-1">— source : Uploaded Pdf</span>
                                          </>
                                        ) : src.page ? (
                                          <>
                                            <span>{`Page ${src.page}`}</span>
                                            <span className="ml-1">— source : Uploaded Pdf</span>
                                          </>
                                        ) : (
                                          <span>source : Uploaded Pdf</span>
                                        )}
                                      </li>
                                    );
                                  })}
                                </ul>
                              )}
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-sm">{message.content}</p>
                      )}
                      <p className="text-xs opacity-70 mt-1">
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="flex justify-start">
              <div className="flex items-start space-x-3 max-w-[80%]">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                  <Bot className="h-4 w-4" />
                </div>
                <ThinkingDots />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {showScrollToTop && (
        <Button
          className="fixed bottom-20 right-6 z-10 rounded-full w-10 h-10 shadow-lg"
          size="icon"
          onClick={scrollToTop}
          title="Scroll to top"
        >
          <ArrowUp className="h-4 w-4" />
        </Button>
      )}
      
      {showScrollToBottom && (
        <Button
          className="fixed bottom-6 right-6 z-10 rounded-full w-10 h-10 shadow-lg"
          size="icon"
          onClick={scrollToBottom}
          title="Scroll to bottom"
        >
          <ArrowDown className="h-4 w-4" />
        </Button>
      )}

      <div className="p-4 border-t">
        <div className="flex space-x-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your document..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button onClick={handleSendMessage} disabled={isLoading || !inputValue.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </Card>
  );
}
