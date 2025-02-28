import { useState, useEffect } from "react";
import { Plus, X, Save } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useGraph } from "@/contexts/GraphContext";
import { apiService, GraphConfig } from "@/services/api";
import { toast } from "sonner";

interface EntityType {
  id: string;
  name: string;
}

export function ConfigurationPage() {
  const { currentGraph, refreshGraphs } = useGraph();

  const [graphName, setGraphName] = useState("");
  const [graphDomain, setGraphDomain] = useState("general");
  const [entityTypes, setEntityTypes] = useState<EntityType[]>([]);
  const [newEntityType, setNewEntityType] = useState("");
  const [exampleQueries, setExampleQueries] = useState("");
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [isSaving, setSaving] = useState(false);

  // Load current graph data when it changes
  useEffect(() => {
    if (currentGraph && !isCreatingNew) {
      setGraphName(currentGraph.name);
      setGraphDomain(currentGraph.domain);
      setEntityTypes(
        currentGraph.entity_types.map((name, index) => ({
          id: index.toString(),
          name,
        }))
      );
      setExampleQueries(currentGraph.example_queries.join("\n"));
    }
  }, [currentGraph, isCreatingNew]);

  const handleAddEntityType = () => {
    if (newEntityType.trim()) {
      setEntityTypes([
        ...entityTypes,
        { id: Date.now().toString(), name: newEntityType.trim() },
      ]);
      setNewEntityType("");
    }
  };

  const handleRemoveEntityType = (id: string) => {
    setEntityTypes(entityTypes.filter((type) => type.id !== id));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddEntityType();
    }
  };

  const handleCreateOrUpdateGraph = async () => {
    try {
      setSaving(true);

      const config: GraphConfig = {
        name: graphName,
        domain: graphDomain,
        entity_types: entityTypes.map((et) => et.name),
        example_queries: exampleQueries.split("\n").filter((q) => q.trim()),
      };

      if (isCreatingNew) {
        await apiService.createGraph(config);
        setIsCreatingNew(false);
        toast.success("New graph created successfully");
      } else if (currentGraph) {
        await apiService.updateGraph(currentGraph.id, config);
        toast.success("Graph configuration updated");
      }

      await refreshGraphs();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to save graph"
      );
    } finally {
      setSaving(false);
    }
  };

  const handleNewGraph = () => {
    setIsCreatingNew(true);
    setGraphName("New Graph");
    setGraphDomain("general");
    setEntityTypes([
      { id: "1", name: "person" },
      { id: "2", name: "place" },
      { id: "3", name: "object" },
      { id: "4", name: "activity" },
      { id: "5", name: "event" },
      { id: "6", name: "organization" },
    ]);
    setExampleQueries("");
  };

  return (
    <>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-1">Configuration</h1>
        <p className="text-gray-400">Configure your graph</p>
      </div>

      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          {isCreatingNew ? (
            <span className="font-medium">Creating New Graph</span>
          ) : currentGraph ? (
            <>
              <div className="h-5 w-5 rounded-full bg-white flex items-center justify-center mr-2">
                <div className="h-2.5 w-2.5 rounded-full bg-black"></div>
              </div>
              <span className="font-medium">{currentGraph.name}</span>
            </>
          ) : (
            <span className="font-medium">No Graph Selected</span>
          )}
        </div>
        <div className="flex gap-2">
          {!isCreatingNew && (
            <Button variant="outline" onClick={handleNewGraph}>
              New Graph
            </Button>
          )}
          <Button
            variant="secondary"
            onClick={handleCreateOrUpdateGraph}
            disabled={isSaving || !graphName.trim()}
          >
            {isSaving ? "Saving..." : "Save"}
            {!isSaving && <Save size={16} className="ml-2" />}
          </Button>
        </div>
      </div>

      <Card className="border-gray-800 bg-[#0f0f13] mb-8">
        <CardContent className="pt-6">
          <div className="mb-6">
            <h2 className="text-xl font-bold mb-1">Graph Name</h2>
            <p className="text-gray-400 text-sm mb-4">
              Give your graph a descriptive name
            </p>
            <input
              type="text"
              value={graphName}
              onChange={(e) => setGraphName(e.target.value)}
              className="w-full p-2 bg-black border border-gray-800 rounded-md focus:outline-none focus:ring-1 focus:ring-gray-700 text-white"
            />
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-bold mb-1">Domain</h2>
            <p className="text-gray-400 text-sm mb-4">
              The domain helps the model understand the context of your data
            </p>
            <input
              type="text"
              value={graphDomain}
              onChange={(e) => setGraphDomain(e.target.value)}
              className="w-full p-2 bg-black border border-gray-800 rounded-md focus:outline-none focus:ring-1 focus:ring-gray-700 text-white"
            />
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-bold mb-1">Entity Types</h2>
            <p className="text-gray-400 text-sm mb-4">
              These are the typologies of entities that are used to generate the
              graph.
            </p>

            <div className="flex flex-wrap gap-2 mb-4">
              {entityTypes.map((type) => (
                <Badge
                  key={type.id}
                  className="bg-gray-800 hover:bg-gray-700 text-white px-3 py-1 rounded-md flex items-center gap-2"
                >
                  {type.name}
                  <button
                    onClick={() => handleRemoveEntityType(type.id)}
                    className="text-white hover:text-gray-200"
                  >
                    <X size={14} />
                  </button>
                </Badge>
              ))}
            </div>

            <div className="relative">
              <input
                type="text"
                value={newEntityType}
                onChange={(e) => setNewEntityType(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Add new entity type"
                className="w-full p-2 pr-10 bg-black border border-gray-800 rounded-md focus:outline-none focus:ring-1 focus:ring-gray-700 text-white"
              />
              <button
                onClick={handleAddEntityType}
                disabled={!newEntityType.trim()}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 w-6 h-6 flex items-center justify-center bg-gray-800 hover:bg-gray-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Plus size={16} className="text-white" />
              </button>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-bold mb-1">Example Queries</h2>
            <p className="text-gray-400 text-sm mb-4">
              These example queries help the model understand how to answer
              questions about your data. Enter one query per line.
            </p>
            <textarea
              value={exampleQueries}
              onChange={(e) => setExampleQueries(e.target.value)}
              placeholder="Enter example queries, one per line"
              className="w-full p-4 bg-black border border-gray-800 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-gray-700 text-white"
              rows={5}
            />
          </div>
        </CardContent>
      </Card>
    </>
  );
}
