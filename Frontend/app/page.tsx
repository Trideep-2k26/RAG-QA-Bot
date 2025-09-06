"use client";

import { useRouter } from "next/navigation";
import { FileUploader } from "@/components/FileUploader";

export default function Home() {
  const router = useRouter();

  const handleUploadComplete = (filename: string) => {
    // Redirect to Q&A page after successful upload
    router.push("/qa");
  };

  return (
    <div className="container mx-auto px-4 py-12 min-h-screen flex items-center justify-center">
      <div className="text-center space-y-8">
        <div className="space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Welcome to PDF Q&A
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Upload your PDF document and start asking questions about its content. 
            Our AI will help you understand and extract information quickly.
          </p>
        </div>
        
        <FileUploader onUploadComplete={handleUploadComplete} />
        
        <div className="max-w-2xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div className="text-center">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                📄
              </div>
              <p className="font-medium">Upload PDF</p>
              <p className="text-muted-foreground">
                Drag & drop or select your PDF file
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                💬
              </div>
              <p className="font-medium">Ask Questions</p>
              <p className="text-muted-foreground">
                Chat with AI about your document
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                ⚡
              </div>
              <p className="font-medium">Get Answers</p>
              <p className="text-muted-foreground">
                Receive instant, accurate responses
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}