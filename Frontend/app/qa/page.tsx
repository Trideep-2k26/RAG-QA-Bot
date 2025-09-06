"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ChatUI } from "@/components/ChatUI";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Upload } from "lucide-react";

export default function QAPage() {
  const [pdfName, setPdfName] = useState<string>("");
  const router = useRouter();

  useEffect(() => {
    // Get uploaded file name from localStorage
    const storedFileName = localStorage.getItem("uploadedFileName");
    if (storedFileName) {
      setPdfName(storedFileName);
    } else {
      // Redirect to home if no file uploaded
      router.push("/");
    }
  }, [router]);

  const handleBackToUpload = () => {
    localStorage.removeItem("uploadedFileName");
    router.push("/");
  };

  if (!pdfName) {
    return (
      <div className="container mx-auto px-4 py-12 min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="sm" onClick={handleBackToUpload}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Upload
          </Button>
          <div>
            <h1 className="text-2xl font-semibold">Q&A Session</h1>
            <p className="text-muted-foreground">
              Currently discussing: <span className="font-medium">{pdfName}</span>
            </p>
          </div>
        </div>
        
        <Button variant="outline" onClick={handleBackToUpload}>
          <Upload className="h-4 w-4 mr-2" />
          Upload New File
        </Button>
      </div>

      {/* Chat Interface */}
      <ChatUI pdfName={pdfName} />

      {/* Tips */}
      <div className="mt-8 max-w-4xl mx-auto">
        <div className="bg-muted/30 rounded-lg p-6">
          <h3 className="font-medium mb-3">💡 Tips for better results:</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Ask specific questions about the document content</li>
            <li>• Reference page numbers or sections when possible</li>
            <li>• Request summaries of complex topics</li>
            <li>• Ask for explanations of technical terms</li>
          </ul>
        </div>
      </div>
    </div>
  );
}