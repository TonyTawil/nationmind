import { useState } from "react";
import {
  Upload,
  Globe,
  Database as DatabaseIcon,
  Youtube,
  MessageSquare,
  FileText,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useGraph } from "@/contexts/GraphContext";
import { apiService } from "@/services/api";
import { toast } from "sonner";

export function DataPage() {
  const { currentGraph, refreshGraphs } = useGraph();
  const [isUploading, setIsUploading] = useState(false);
  const [textContent, setTextContent] = useState("");
  const [showTextInput, setShowTextInput] = useState(false);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!currentGraph) {
      toast.error("No graph selected. Please create or select a graph first.");
      return;
    }

    const files = event.target.files;
    if (!files || files.length === 0) return;

    try {
      setIsUploading(true);

      // Upload each file
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        await apiService.uploadFile(currentGraph.id, file);

        toast.success(`File "${file.name}" uploaded successfully`);
      }

      // Refresh graph data to update counts
      await refreshGraphs();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to upload file"
      );
    } finally {
      setIsUploading(false);
      // Reset the input
      event.target.value = "";
    }
  };

  const handleTextUpload = async () => {
    if (!currentGraph) {
      toast.error("No graph selected. Please create or select a graph first.");
      return;
    }

    if (!textContent.trim()) {
      toast.error("Please enter some text content");
      return;
    }

    try {
      setIsUploading(true);
      await apiService.uploadText(currentGraph.id, textContent);

      toast.success("Text content uploaded successfully");

      // Reset and hide text input
      setTextContent("");
      setShowTextInput(false);

      // Refresh graph data to update counts
      await refreshGraphs();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to upload text"
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-1">Data</h1>
        <p className="text-gray-400">Add and manage the data in your graph.</p>
      </div>

      {!currentGraph ? (
        <div className="text-center py-8">
          <p>No graphs available. Create a graph in the Configuration page.</p>
        </div>
      ) : (
        <>
          {showTextInput && (
            <Card className="border-gray-800 bg-[#0f0f13] mb-6">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <FileText className="mr-2" size={20} />
                  <CardTitle>Enter Text Content</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <textarea
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  placeholder="Enter text content to add to your graph..."
                  className="w-full p-4 bg-black border border-gray-800 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-gray-700 text-white"
                  rows={8}
                />
                <div className="flex gap-2 mt-4">
                  <Button
                    onClick={handleTextUpload}
                    disabled={isUploading || !textContent.trim()}
                  >
                    {isUploading ? "Uploading..." : "Upload Text"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setShowTextInput(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* First row of data source cards */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            {/* Upload File(s) card */}
            <Card className="border-gray-800 bg-[#0f0f13] flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <Upload className="mr-2" size={20} />
                  <CardTitle>Upload File(s)</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <p className="text-gray-400 text-sm mb-4">
                  Upload your local files directly
                </p>

                <div className="mb-6">
                  <h3 className="font-medium mb-2">Features:</h3>
                  <ul className="text-gray-400 text-sm space-y-1">
                    <li>• Support for PDFs, TXTs, and MDs</li>
                    <li>• Batch upload</li>
                    <li>• Automatic file parsing</li>
                  </ul>
                </div>

                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  multiple
                  accept=".pdf,.txt,.md"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
                <label htmlFor="file-upload">
                  <Button
                    className="mt-auto w-full"
                    variant="secondary"
                    disabled={isUploading}
                    onClick={() =>
                      document.getElementById("file-upload")?.click()
                    }
                  >
                    {isUploading ? "Uploading..." : "Browse"}
                  </Button>
                </label>
              </CardContent>
            </Card>

            {/* Text Input card */}
            <Card className="border-gray-800 bg-[#0f0f13] flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <FileText className="mr-2" size={20} />
                  <CardTitle>Text Input</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <p className="text-gray-400 text-sm mb-4">
                  Enter text content directly
                </p>

                <div className="mb-6">
                  <h3 className="font-medium mb-2">Features:</h3>
                  <ul className="text-gray-400 text-sm space-y-1">
                    <li>• Direct text entry</li>
                    <li>• Paste from clipboard</li>
                    <li>• Automatic processing</li>
                  </ul>
                </div>

                <Button
                  className="mt-auto w-full"
                  variant="secondary"
                  onClick={() => setShowTextInput(true)}
                >
                  Enter Text
                </Button>
              </CardContent>
            </Card>

            {/* Web Crawler card */}
            <Card className="border-gray-800 bg-[#0f0f13] flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <Globe className="mr-2" size={20} />
                  <CardTitle>Web Crawler</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <p className="text-gray-400 text-sm mb-4">
                  Crawl websites for data extraction
                </p>

                <div className="mb-6">
                  <h3 className="font-medium mb-2">Features:</h3>
                  <ul className="text-gray-400 text-sm space-y-1">
                    <li>• Customizable crawling rules</li>
                    <li>• Scheduled crawls</li>
                    <li>• Data extraction templates</li>
                  </ul>
                </div>

                <Button className="mt-auto w-full" variant="secondary">
                  Manage
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Second row of data source cards */}
          <div className="grid grid-cols-3 gap-4">
            {/* Google Cloud Storage card */}
            <Card className="border-gray-800 bg-[#0f0f13] flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <DatabaseIcon className="mr-2" size={20} />
                  <CardTitle>Google Cloud Storage</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <p className="text-gray-400 text-sm mb-4">
                  Integrate with Google Cloud Storage
                </p>

                <div className="mb-6">
                  <h3 className="font-medium mb-2">Features:</h3>
                  <ul className="text-gray-400 text-sm space-y-1">
                    <li>• High-performance object storage</li>
                    <li>• Global data access</li>
                    <li>• Strong consistency</li>
                  </ul>
                </div>

                <Button className="mt-auto w-full" variant="secondary" disabled>
                  Enterprise Plan Only
                </Button>
              </CardContent>
            </Card>

            {/* YouTube card */}
            <Card className="border-gray-800 bg-[#0f0f13] flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <Youtube className="mr-2" size={20} />
                  <CardTitle>YouTube</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <p className="text-gray-400 text-sm mb-4">
                  Extract data from YouTube videos
                </p>

                <div className="mb-6">
                  <h3 className="font-medium mb-2">Features:</h3>
                  <ul className="text-gray-400 text-sm space-y-1">
                    <li>• Video metadata extraction</li>
                    <li>• Transcript analysis</li>
                    <li>• Comment data mining</li>
                  </ul>
                </div>

                <Button className="mt-auto w-full" variant="secondary" disabled>
                  Enterprise Plan Only
                </Button>
              </CardContent>
            </Card>

            {/* Slack card */}
            <Card className="border-gray-800 bg-[#0f0f13] flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center">
                  <MessageSquare className="mr-2" size={20} />
                  <CardTitle>Slack</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <p className="text-gray-400 text-sm mb-4">
                  Connect to your Slack workspace
                </p>

                <div className="mb-6">
                  <h3 className="font-medium mb-2">Features:</h3>
                  <ul className="text-gray-400 text-sm space-y-1">
                    <li>• Channel history import</li>
                    <li>• Real-time message streaming</li>
                    <li>• User data integration</li>
                  </ul>
                </div>

                <Button className="mt-auto w-full" variant="secondary" disabled>
                  Enterprise Plan Only
                </Button>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
