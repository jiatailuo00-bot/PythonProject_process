export interface ScriptParameter {
  name: string;
  label: string;
  type: 'string' | 'number' | 'boolean' | 'path' | 'select';
  required?: boolean;
  description?: string;
  placeholder?: string;
  options?: string[];
  example?: unknown;
}

export interface ScriptMetadata {
  id: string;
  name: string;
  description: string;
  category: string;
  parameters: ScriptParameter[];
  doc_url?: string;
  output_description?: string;
}

export interface ScriptRunResponse {
  success: boolean;
  message: string;
  data: Record<string, unknown>;
  logs?: string;
}

export interface UploadResult {
  filename: string;
  path: string;
  size: number;
  success?: boolean;
  message?: string;
  error?: string;
}

export interface BatchUploadResponse {
  total: number;
  successful: number;
  failed: number;
  results: UploadResult[];
  message: string;
}

export interface UploadedFile {
  filename: string;
  path: string;
  size: number;
  created_time: string;
}
