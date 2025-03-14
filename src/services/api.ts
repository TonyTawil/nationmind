// API service for Circlemind

// Use relative URLs instead of hardcoded localhost
const API_BASE_URL = '';

// Types based on backend models
export interface GraphConfig {
  name: string;
  domain: string;
  entity_types: string[];
  example_queries?: string[];
}

export interface GraphInfo {
  id: string;
  name: string;
  domain: string;
  entity_types: string[];
  example_queries: string[];
  node_count: number;
  edge_count: number;
  chunk_count: number;
  created_at: string;
  last_updated: string;
}

export interface GraphQueryRequest {
  query: string;
  with_references: boolean;
}

export interface GraphQueryResponse {
  query: string;
  response: string;
  context?: any;
}

export interface Entity {
  index: number;
  id: string;
  name: string;
  type: string;
  description: string;
}

export interface Relationship {
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

export interface ExplorerData {
  entities: Entity[];
  relationships: Relationship[];
  chunks: string[];
}

// API functions
export const apiService = {
  // Graph management
  async createGraph(config: GraphConfig): Promise<{ graph_id: string; message: string }> {
    const response = await fetch(`${API_BASE_URL}/graphs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create graph: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async listGraphs(): Promise<GraphInfo[]> {
    const response = await fetch(`${API_BASE_URL}/graphs`);
    
    if (!response.ok) {
      throw new Error(`Failed to list graphs: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async getGraphInfo(graphId: string): Promise<GraphInfo> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get graph info: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async updateGraph(graphId: string, config: GraphConfig): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update graph: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async deleteGraph(graphId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`Failed to delete graph: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // Data ingestion
  async uploadText(graphId: string, content: string, filename?: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/upload-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, filename }),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to upload text: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async uploadFile(graphId: string, file: File): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/upload-file`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Failed to upload file: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // Querying
  async queryGraph(graphId: string, queryRequest: GraphQueryRequest): Promise<GraphQueryResponse> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(queryRequest),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to query graph: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // Explorer data
  async getExplorerData(graphId: string): Promise<ExplorerData> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/explorer`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch explorer data: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // Graph diagnostics
  async diagnoseGraph(graphId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/diagnose`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Failed to diagnose graph: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // Refresh graph counts
  async refreshGraphCounts(graphId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/refresh-counts`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Failed to refresh graph counts: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // Reload graph storage
  async reloadGraphStorage(graphId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/graphs/${graphId}/reload-storage`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Failed to reload graph storage: ${response.statusText}`);
    }
    
    return response.json();
  }
}; 