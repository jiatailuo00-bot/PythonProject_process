import axios from 'axios';
import type {
  BatchUploadResponse,
  ScriptMetadata,
  ScriptRunResponse,
  UploadResult,
  UploadedFile,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

const client = axios.create({
  baseURL: API_BASE_URL,
});

const unwrap = <T>(promise: Promise<{ data: T }>) => promise.then((res) => res.data);

export function fetchScripts() {
  return unwrap(client.get<ScriptMetadata[]>('/scripts'));
}

export function runScript(scriptId: string, params: Record<string, unknown>) {
  return unwrap(
    client.post<ScriptRunResponse>(`/scripts/${scriptId}/run`, {
      params,
    }),
  );
}

export function uploadSingleFile(file: File) {
  const form = new FormData();
  form.append('file', file);
  return unwrap(client.post<UploadResult>('/upload/single', form));
}

export function uploadBatchFiles(files: File[]) {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  return unwrap(client.post<BatchUploadResponse>('/upload/batch', form));
}

export function listUploadedFiles() {
  return unwrap(client.get<{ files: UploadedFile[]; total: number }>('/upload/list'));
}

export function deleteUploadedFile(filename: string) {
  return unwrap(client.delete<{ message: string }>(`/upload/${encodeURIComponent(filename)}`));
}

export function deleteUploadedFiles(filenames: string[]) {
  return unwrap(
    client.post<{ requested: number; deleted: string[]; failed: { filename: string; error: string }[] }>(
      '/upload/batch-delete',
      { filenames },
    ),
  );
}
