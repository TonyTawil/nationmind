import { useState, useEffect } from "react";
import { Search, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useGraph } from "@/contexts/GraphContext";
import { toast } from "sonner";

// Types for our explorer data
interface Entity {
  index: number;
  id: string;
  name: string;
  type: string;
  description: string;
}

interface Relationship {
  index: number;
  source: string;
  target: string;
  source_id: string;
  target_id: string;
  source_type: string;
  target_type: string;
  description: string;
  predicate: string;
  chunks: string[];
}

export function KnowledgeExplorerPage() {
  const { currentGraph } = useGraph();
  const [searchQuery, setSearchQuery] = useState("");

  // State for explorer data
  const [entities, setEntities] = useState<Entity[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [chunks, setChunks] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // State for expanded items
  const [expandedEntities, setExpandedEntities] = useState<
    Record<number, boolean>
  >({});
  const [expandedRelationships, setExpandedRelationships] = useState<
    Record<number, boolean>
  >({});

  // Function to toggle entity expansion
  const toggleEntity = (index: number) => {
    setExpandedEntities((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  // Function to toggle relationship expansion
  const toggleRelationship = (index: number) => {
    setExpandedRelationships((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  // Function to fetch explorer data
  const fetchExplorerData = async () => {
    if (!currentGraph) return;

    setIsLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/graphs/${currentGraph.id}/explorer`
      );
      if (!response.ok) {
        throw new Error(
          `Failed to fetch explorer data: ${response.statusText}`
        );
      }

      const data = await response.json();
      setEntities(data.entities || []);
      setRelationships(data.relationships || []);
      setChunks(data.chunks || []);
    } catch (error) {
      console.error("Error fetching explorer data:", error);
      toast.error("Failed to load graph data");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch data when the component mounts or when the current graph changes
  useEffect(() => {
    fetchExplorerData();
  }, [currentGraph]);

  // Filter entities and relationships based on search query
  const filteredEntities = entities.filter(
    (entity) =>
      searchQuery === "" ||
      entity.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entity.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entity.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredRelationships = relationships.filter(
    (rel) =>
      searchQuery === "" ||
      rel.source.toLowerCase().includes(searchQuery.toLowerCase()) ||
      rel.target.toLowerCase().includes(searchQuery.toLowerCase()) ||
      rel.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-1">Knowledge Explorer</h1>
        <p className="text-gray-400">
          Explore your graph to understand how the model thinks
        </p>
      </div>

      {!currentGraph ? (
        <div className="text-center py-8">
          <p>No graphs available. Create a graph in the Configuration page.</p>
        </div>
      ) : (
        <>
          {/* Graph visualization area */}
          <Card className="border-gray-800 bg-[#0f0f13] mb-6 relative">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="flex items-center">
                <div className="h-5 w-5 rounded-full bg-white flex items-center justify-center mr-2">
                  <div className="h-2.5 w-2.5 rounded-full bg-black"></div>
                </div>
                <CardTitle>Graph - Interactive</CardTitle>
              </div>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search
                    className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                    size={16}
                  />
                  <input
                    type="text"
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-black border border-gray-800 rounded-md text-sm w-64 focus:outline-none focus:ring-1 focus:ring-gray-700"
                  />
                </div>
                <Button
                  variant="outline"
                  size="icon"
                  className="rounded-md"
                  onClick={fetchExplorerData}
                  disabled={isLoading}
                >
                  <RefreshCw
                    size={16}
                    className={isLoading ? "animate-spin" : ""}
                  />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0 h-[500px] relative">
              {/* This would be where your graph visualization component goes */}
              <div className="absolute inset-0 flex items-center justify-center">
                {/* Placeholder for graph visualization */}
                <div className="text-gray-500">
                  Graph visualization would render here
                </div>
              </div>

              {/* Sample node labels that would be positioned by the graph visualization */}
              <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                Showing {entities.length} nodes and {relationships.length} edges
              </div>
            </CardContent>
          </Card>

          {/* Bottom panels */}
          <div className="grid grid-cols-3 gap-4">
            {/* Entities panel */}
            <Card className="border-gray-800 bg-[#0f0f13]">
              <CardHeader className="pb-2">
                <CardTitle>Entities</CardTitle>
              </CardHeader>
              <CardContent
                className="max-h-[400px] overflow-auto pr-2"
                style={{ scrollbarWidth: "none" }}
              >
                {isLoading ? (
                  <div className="text-center py-4">Loading entities...</div>
                ) : filteredEntities.length > 0 ? (
                  filteredEntities.map((entity, index) => (
                    <div key={index} className="py-2 border-b border-gray-800">
                      <div className="flex justify-between items-center">
                        <div className="font-medium">{entity.name}</div>
                        <div className="flex items-center">
                          <span className="text-xs text-gray-400 mr-2">
                            {entity.type}
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => toggleEntity(entity.index)}
                          >
                            {expandedEntities[entity.index] ? (
                              <ChevronUp size={15} />
                            ) : (
                              <ChevronDown size={15} />
                            )}
                          </Button>
                        </div>
                      </div>

                      {expandedEntities[entity.index] && (
                        <div className="mt-2 pl-2 text-sm text-gray-400 space-y-1">
                          <div>
                            <span className="font-medium text-gray-300">
                              Name:
                            </span>{" "}
                            {entity.name}
                          </div>
                          <div>
                            <span className="font-medium text-gray-300">
                              Type:
                            </span>{" "}
                            {entity.type}
                          </div>
                          <div>
                            <span className="font-medium text-gray-300">
                              Description:
                            </span>{" "}
                            {entity.description}
                          </div>
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4">No entities found</div>
                )}
              </CardContent>
            </Card>

            {/* Relationships panel */}
            <Card className="border-gray-800 bg-[#0f0f13]">
              <CardHeader className="pb-2">
                <CardTitle>Relationships</CardTitle>
              </CardHeader>
              <CardContent
                className="max-h-[400px] overflow-auto pr-2"
                style={{ scrollbarWidth: "none" }}
              >
                {isLoading ? (
                  <div className="text-center py-4">
                    Loading relationships...
                  </div>
                ) : filteredRelationships.length > 0 ? (
                  filteredRelationships.map((rel, index) => (
                    <div className="py-2 border-b border-gray-800" key={index}>
                      <div className="flex justify-between items-center mb-1">
                        <div className="font-medium">
                          [{rel.source} → {rel.target}]
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => toggleRelationship(rel.index)}
                        >
                          {expandedRelationships[rel.index] ? (
                            <ChevronUp size={15} />
                          ) : (
                            <ChevronDown size={15} />
                          )}
                        </Button>
                      </div>

                      {!expandedRelationships[rel.index] ? (
                        <p className="text-sm text-gray-400">
                          {rel.description}
                        </p>
                      ) : (
                        <div className="mt-2 pl-2 text-sm text-gray-400 space-y-1">
                          <div>
                            <span className="font-medium text-gray-300">
                              Subject:
                            </span>{" "}
                            {rel.source}{" "}
                            <span className="text-xs">({rel.source_type})</span>
                          </div>
                          <div>
                            <span className="font-medium text-gray-300">
                              Predicate:
                            </span>{" "}
                            {rel.predicate}
                          </div>
                          <div>
                            <span className="font-medium text-gray-300">
                              Object:
                            </span>{" "}
                            {rel.target}{" "}
                            <span className="text-xs">({rel.target_type})</span>
                          </div>
                          {rel.chunks.length > 0 && (
                            <div>
                              <span className="font-medium text-gray-300">
                                Chunks:
                              </span>{" "}
                              {rel.chunks.length}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4">No relationships found</div>
                )}
              </CardContent>
            </Card>

            {/* Chunks panel */}
            <Card className="border-gray-800 bg-[#0f0f13]">
              <CardHeader className="pb-2">
                <CardTitle>Chunks</CardTitle>
              </CardHeader>
              <CardContent
                className="max-h-[400px] overflow-auto pr-2"
                style={{ scrollbarWidth: "none" }}
              >
                {isLoading ? (
                  <div className="text-center py-4">Loading chunks...</div>
                ) : chunks.length > 0 ? (
                  chunks.map((chunkId, index) => (
                    <div
                      key={index}
                      className="text-xs text-gray-400 break-all py-2 border-b border-gray-800"
                    >
                      {chunkId}
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4">No chunks found</div>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
