"use client";

import { useState, useRef, DragEvent } from "react";
import { Upload, File as FileIcon, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";
import { API_CONFIG } from "@/lib/api-config";

interface FileUploaderProps {
  onUploadComplete: (filename: string) => void;
}

export function FileUploader({ onUploadComplete }: FileUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ALLOWED_EXTS = [".pdf", ".docx", ".csv", ".xlsx", ".xls", ".png"] as const;
  const isAllowedFile = (file: File | null | undefined): file is File => {
    if (!file) return false;
    const name = (file.name || '').toLowerCase();
    const extOk = ALLOWED_EXTS.some((ext) => name.endsWith(ext));
    if (extOk) return true;
    const ct = (file.type || '').toLowerCase();
    const mimeOk = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'image/png',
      'application/octet-stream',
    ].includes(ct);
    return mimeOk;
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    const picked = files.find((file) => isAllowedFile(file));

    if (picked) {
      setSelectedFile(picked);
    } else {
      toast.error("Please select a supported file: PDF, DOCX, CSV, XLSX/XLS, PNG");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (isAllowedFile(file)) {
      setSelectedFile(file!);
    } else {
      toast.error("Please select a supported file: PDF, DOCX, CSV, XLSX/XLS, PNG");
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error("No file selected. Please choose a supported file.");
      return;
    }

    if (!isAllowedFile(selectedFile)) {
      toast.error("Invalid file type. Please select PDF, DOCX, CSV, XLSX/XLS, or PNG.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      toast.error("File too large. Maximum size is 10MB.");
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      
      const useTestEndpoint = process.env.NEXT_PUBLIC_USE_TEST_UPLOAD === 'true';
      const endpoint = useTestEndpoint ? "test-upload" : "upload";
      const uploadUrl = API_CONFIG.getUrl(endpoint);

      console.log("Upload details:", {
        url: uploadUrl,
        fileType: selectedFile.type,
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
      });

      const response = await fetch(uploadUrl, {
        method: "POST",
        headers: {
          'Accept': 'application/json',
        },
        body: formData,
        credentials: 'omit',
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Upload successful:", data);

        localStorage.setItem("uploadedFileName", selectedFile.name);
        toast.success("File uploaded successfully!");
        onUploadComplete(selectedFile.name);
      } else {
        let errorMessage = "Upload failed";
        try {
          const errorData = await response.json();
          console.error("Upload failed:", errorData);
          errorMessage = errorData.detail || JSON.stringify(errorData);
        } catch (e) {
          const errorText = await response.text();
          console.error("Upload failed:", errorText);
          errorMessage = errorText;
        }
        toast.error(`Upload failed: ${errorMessage}`);
      }
    } catch (error: any) {
      console.error("Upload error:", error);
      const errorMessage = error.message || "Failed to upload file";
      toast.error(`Error: ${errorMessage}. Please try again.`);
    } finally {
      setIsUploading(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <Card className="p-8 w-full max-w-2xl mx-auto">
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-2">Upload Your Document</h2>
          <p className="text-muted-foreground">
            Upload a document to start asking questions about its content
          </p>
        </div>

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center ${
            isDragOver ? "border-primary bg-primary/10" : "border-muted"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {selectedFile ? (
            <div className="space-y-4">
              <div className="flex items-center justify-center space-x-2">
                <FileIcon className="h-8 w-8" />
                <span className="text-lg font-medium">{selectedFile.name}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full"
                  onClick={clearFile}
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
              <Button onClick={handleUpload} disabled={isUploading}>
                {isUploading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  "Upload File"
                )}
              </Button>
            </div>
          ) : (
            <>
              <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <div className="space-y-2">
                <p>Drag and drop your document here, or click to select</p>
                <p className="text-xs text-muted-foreground">
                  Supports PDF, DOCX, CSV, XLSX/XLS, PNG up to 10MB
                </p>
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose File
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.csv,.xlsx,.xls,.png,application/pdf,application/x-pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,image/png"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            </>
          )}
        </div>
      </div>
    </Card>
  );
}