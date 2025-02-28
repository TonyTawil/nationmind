import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import { apiService, GraphInfo } from "@/services/api";

interface GraphContextType {
  graphs: GraphInfo[];
  currentGraph: GraphInfo | null;
  loading: boolean;
  error: string | null;
  refreshGraphs: () => Promise<void>;
  setCurrentGraphById: (id: string) => void;
}

const GraphContext = createContext<GraphContextType | undefined>(undefined);

export function GraphProvider({ children }: { children: ReactNode }) {
  const [graphs, setGraphs] = useState<GraphInfo[]>([]);
  const [currentGraph, setCurrentGraph] = useState<GraphInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshGraphs = async () => {
    try {
      setLoading(true);
      setError(null);
      const fetchedGraphs = await apiService.listGraphs();
      setGraphs(fetchedGraphs);

      // If we have a current graph, refresh its data
      if (currentGraph) {
        const updatedGraph = fetchedGraphs.find(
          (g) => g.id === currentGraph.id
        );
        if (updatedGraph) {
          setCurrentGraph(updatedGraph);
        }
      }
      // If no current graph but we have graphs, set the first one as current
      else if (fetchedGraphs.length > 0 && !currentGraph) {
        setCurrentGraph(fetchedGraphs[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch graphs");
    } finally {
      setLoading(false);
    }
  };

  const setCurrentGraphById = (id: string) => {
    const graph = graphs.find((g) => g.id === id);
    if (graph) {
      setCurrentGraph(graph);
    }
  };

  // Initial load
  useEffect(() => {
    refreshGraphs();
  }, []);

  return (
    <GraphContext.Provider
      value={{
        graphs,
        currentGraph,
        loading,
        error,
        refreshGraphs,
        setCurrentGraphById,
      }}
    >
      {children}
    </GraphContext.Provider>
  );
}

export function useGraph() {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error("useGraph must be used within a GraphProvider");
  }
  return context;
}
