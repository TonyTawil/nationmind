import { useState, useRef, useEffect } from "react";
import { Send, RefreshCw, FileText, Link as LinkIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useGraph } from "@/contexts/GraphContext";
import { apiService } from "@/services/api";
import { toast } from "sonner";
import { Separator } from "@/components/ui/separator";

interface Message {
  id: string;
  type: "system" | "user" | "assistant";
  content: string;
  references?: Reference[];
}

interface Reference {
  id: string;
  text: string;
  metadata?: Record<string, any>;
}

export function ChatPage() {
  const { currentGraph } = useGraph();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "system",
      content:
        "Welcome to the debugger 👋 Ask a question to find and fix issues in your Knowledge Graph.",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedReference, setSelectedReference] = useState<Reference | null>(
    null
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    if (!currentGraph) {
      toast.error("No graph selected. Please create or select a graph first.");
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      // Send query to API
      const response = await apiService.queryGraph(currentGraph.id, {
        query: userMessage.content,
        with_references: true,
      });

      // Process references if they exist
      let references: Reference[] = [];
      if (response.context && response.context.references) {
        references = response.context.references.map((ref: any) => ({
          id: ref.id || Math.random().toString(),
          text: ref.text || "",
          metadata: ref.metadata || {},
        }));
      } else if (response.context && response.context.raw_context) {
        // If we have raw context but no structured references
        references = [
          {
            id: Math.random().toString(),
            text: response.context.raw_context,
            metadata: { filename: "Context" },
          },
        ];
      }

      // Add assistant response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: response.response,
        references,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to get response from the graph"
      );

      // Add error message
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: "system",
          content: "Sorry, there was an error processing your request.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Format message content with line breaks
  const formatMessageContent = (content: string) => {
    return content.split("\n").map((line, i) => (
      <span key={i}>
        {line}
        <br />
      </span>
    ));
  };

  return (
    <>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-1">Chat</h1>
        <p className="text-gray-400">
          Ask questions about your knowledge graph
        </p>
      </div>

      {!currentGraph ? (
        <div className="text-center py-8">
          <p>No graphs available. Create a graph in the Configuration page.</p>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-4">
          {/* Chat panel */}
          <Card className="border-gray-800 bg-[#0f0f13] col-span-2">
            <CardHeader className="pb-2">
              <CardTitle>Chat with {currentGraph.name}</CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex flex-col h-[calc(100vh-220px)]">
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.type === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.type === "user"
                          ? "bg-blue-600 text-white"
                          : message.type === "system"
                          ? "bg-gray-700 text-white"
                          : "bg-gray-800 text-white"
                      }`}
                    >
                      <div className="text-sm">
                        {formatMessageContent(message.content)}
                      </div>

                      {message.references && message.references.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-gray-700">
                          <div className="text-xs text-gray-400 mb-1">
                            References:
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {message.references.map((ref) => (
                              <Button
                                key={ref.id}
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-xs flex items-center gap-1"
                                onClick={() => setSelectedReference(ref)}
                              >
                                <FileText size={12} />
                                {ref.metadata?.filename || "Source"}
                              </Button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-4 border-t border-gray-800 relative">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question..."
                  className="w-full p-3 pr-12 bg-black border border-gray-800 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-gray-700 text-white"
                  rows={3}
                  disabled={isLoading}
                />
                <Button
                  onClick={handleSendMessage}
                  className="absolute right-6 bottom-6"
                  size="icon"
                  disabled={!inputValue.trim() || isLoading}
                >
                  {isLoading ? (
                    <RefreshCw size={16} className="animate-spin" />
                  ) : (
                    <Send size={16} />
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Reference panel */}
          <Card className="border-gray-800 bg-[#0f0f13] col-span-1">
            <CardHeader className="pb-2">
              <CardTitle>References</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 h-[calc(100vh-220px)] overflow-y-auto">
              {selectedReference ? (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <FileText size={16} className="text-gray-400" />
                      <span className="font-medium">
                        {selectedReference.metadata?.filename || "Source"}
                      </span>
                    </div>
                    {selectedReference.metadata?.url && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0"
                        asChild
                      >
                        <a
                          href={selectedReference.metadata.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <LinkIcon size={14} />
                        </a>
                      </Button>
                    )}
                  </div>
                  <Separator className="my-2" />
                  <div className="text-sm text-gray-300 whitespace-pre-wrap">
                    {selectedReference.text}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  Select a reference to view its content
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
}
