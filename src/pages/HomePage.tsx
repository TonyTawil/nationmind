import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useGraph } from "@/contexts/GraphContext";
import { apiService, GraphQueryResponse } from "@/services/api";

interface RecentQuery {
  query: string;
  response: string;
  timestamp: string;
}

export function HomePage() {
  const { currentGraph, loading } = useGraph();
  const [recentQueries, setRecentQueries] = useState<RecentQuery[]>([]);

  // In a real app, you might fetch recent queries from an endpoint
  // For now, we'll just simulate this with empty data

  return (
    <>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-1">Home</h1>
        <p className="text-gray-400">Monitor the activity on your graph</p>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading graph data...</div>
      ) : !currentGraph ? (
        <div className="text-center py-8">
          <p>No graphs available. Create a graph in the Configuration page.</p>
        </div>
      ) : (
        <>
          {/* Stats cards */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <Card className="border-gray-800 bg-[#0f0f13]">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-gray-400">
                  Entities
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">{currentGraph.node_count}</p>
              </CardContent>
            </Card>
            <Card className="border-gray-800 bg-[#0f0f13]">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-gray-400">Triples</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">{currentGraph.edge_count}</p>
              </CardContent>
            </Card>
            <Card className="border-gray-800 bg-[#0f0f13]">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-gray-400">Chunks</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">{currentGraph.chunk_count}</p>
              </CardContent>
            </Card>
          </div>

          {/* Recent queries */}
          <Card className="border-gray-800 bg-[#0f0f13]">
            <CardHeader className="pb-2">
              <div className="flex items-center">
                <div className="h-5 w-5 rounded-full bg-white flex items-center justify-center mr-2">
                  <div className="h-2.5 w-2.5 rounded-full bg-black"></div>
                </div>
                <CardTitle>{currentGraph.name}</CardTitle>
              </div>
              <p className="text-sm text-gray-400 mt-2">
                List of recent queries to your graph
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 text-sm text-gray-400 border-b border-gray-800 pb-2">
                <div>Query</div>
                <div>Response</div>
                <div>Timestamp</div>
              </div>

              {recentQueries.length === 0 ? (
                <div className="text-center py-4 text-gray-400">
                  No recent queries. Try asking a question in the Chat page.
                </div>
              ) : (
                recentQueries.map((q, i) => (
                  <div
                    key={i}
                    className="grid grid-cols-3 gap-4 py-2 border-b border-gray-800"
                  >
                    <div className="truncate">{q.query}</div>
                    <div className="truncate">{q.response}</div>
                    <div>{q.timestamp}</div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </>
      )}
    </>
  );
}
