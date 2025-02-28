import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { HomePage } from "./pages/HomePage";
import { KnowledgeExplorerPage } from "./pages/KnowledgeExplorerPage";
import { ChatPage } from "./pages/ChatPage";
import { DataPage } from "./pages/DataPage";
import { ConfigurationPage } from "./pages/ConfigurationPage";
import { GraphProvider } from "./contexts/GraphContext";
import { ToastProvider } from "./components/providers/ToastProvider";

const App = () => {
  return (
    <GraphProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route
              path="/knowledge-explorer"
              element={<KnowledgeExplorerPage />}
            />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/configuration" element={<ConfigurationPage />} />
            <Route path="*" element={<HomePage />} />
          </Routes>
        </Layout>
      </Router>
      <ToastProvider />
    </GraphProvider>
  );
};

export default App;
