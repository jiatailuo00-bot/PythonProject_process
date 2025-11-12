<script setup lang="ts">
import { ElMessage } from 'element-plus';
import { onMounted, onUnmounted, ref } from 'vue';
import { uploadBatchFiles, uploadSingleFile } from '../api';
import type { UploadResult } from '../types';

const props = withDefaults(defineProps<{
  multiple?: boolean;
  accept?: string;
  maxSize?: number;
}>(), {
  multiple: false,
  accept: '*/*',
  maxSize: 500 * 1024 * 1024,
});

const emit = defineEmits<{
  (event: 'uploaded', result: UploadResult): void;
  (event: 'batch', results: UploadResult[]): void;
}>();

const isDragging = ref(false);
const uploading = ref(false);
const uploadAreaRef = ref<HTMLElement | null>(null);
let boundArea: HTMLElement | null = null;
const fileInputRef = ref<HTMLInputElement | null>(null);

const formatSize = (bytes: number) => {
  if (!bytes) return '0B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const val = bytes / 1024 ** i;
  return `${val.toFixed(1)}${units[i]}`;
};

const validateFile = (file: File): string | null => {
  if (file.size > props.maxSize) {
    return `文件超过限制 (${formatSize(props.maxSize)})`;
  }
  if (props.accept && props.accept !== '*/*') {
    const allowList = props.accept.split(',').map((item) => item.trim());
    const matched = allowList.some((item) => {
      if (item.startsWith('.')) {
        return file.name.toLowerCase().endsWith(item.toLowerCase());
      }
      const pattern = item.replace('*', '.*');
      return new RegExp(pattern).test(file.type);
    });
    if (!matched) {
      return `不支持的类型: ${file.type || '未知'}`;
    }
  }
  return null;
};

const processFiles = async (files: FileList) => {
  if (!files.length) return;
  uploading.value = true;
  try {
    if (files.length === 1) {
      const file = files[0];
      const validation = validateFile(file);
      if (validation) {
        ElMessage.error(validation);
        return;
      }
      const result = await uploadSingleFile(file);
      emit('uploaded', result);
      ElMessage.success('文件上传成功');
    } else {
      const validFiles: File[] = [];
      const errors: string[] = [];
      Array.from(files).forEach((file) => {
        const validation = validateFile(file);
        if (validation) {
          errors.push(`${file.name}: ${validation}`);
        } else {
          validFiles.push(file);
        }
      });
      if (errors.length) {
        ElMessage.warning(errors.join('\n'));
      }
      if (validFiles.length) {
        const batch = await uploadBatchFiles(validFiles);
        const successful = batch.results.filter((item) => item.success !== false);
        emit('batch', successful);
        ElMessage.success(batch.message);
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : '上传失败';
    ElMessage.error(message);
  } finally {
    uploading.value = false;
  }
};

const handleInputChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files?.length) {
    void processFiles(target.files);
  }
  target.value = '';
};

const handleDrop = (event: DragEvent) => {
  event.preventDefault();
  isDragging.value = false;
  if (event.dataTransfer?.files?.length) {
    void processFiles(event.dataTransfer.files);
  }
};

const handlePaste = (event: ClipboardEvent) => {
  if (!event.clipboardData) return;
  const items = Array.from(event.clipboardData.items).filter((item) => item.kind === 'file');
  if (!items.length) return;
  const files = items
    .map((item) => item.getAsFile())
    .filter((file): file is File => Boolean(file));
  if (files.length) {
    event.preventDefault();
    const dataTransfer = new DataTransfer();
    files.forEach((file) => dataTransfer.items.add(file));
    void processFiles(dataTransfer.files);
  }
};

onMounted(() => {
  boundArea = uploadAreaRef.value;
  boundArea?.addEventListener('paste', handlePaste);
});

onUnmounted(() => {
  boundArea?.removeEventListener('paste', handlePaste);
  boundArea = null;
});
</script>

<template>
  <div class="file-upload">
    <div
      ref="uploadAreaRef"
      class="upload-area"
      :class="{ dragging: isDragging, uploading }"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop="handleDrop"
      @click="fileInputRef?.click()"
    >
      <input
        ref="fileInputRef"
        type="file"
        :multiple="multiple"
        :accept="accept"
        style="display: none"
        @change="handleInputChange"
      />
      <div v-if="uploading">
        <div class="upload-icon uploading">⏳</div>
        <div class="upload-title">正在上传...</div>
        <p class="upload-desc">请稍候，文件处理中</p>
      </div>
      <div v-else>
        <div class="upload-icon">📁</div>
        <div class="upload-title">
          {{ multiple ? '拖拽多个Excel文件' : '拖拽Excel文件' }} 到此处
        </div>
        <p class="upload-desc">或者点击选择文件，支持粘贴 (Ctrl+V)</p>
        <small class="upload-info">
          支持 {{ accept.replace(/\./g, '').toUpperCase() }} 格式 · 最大 {{ formatSize(maxSize) }}
        </small>
      </div>
    </div>
  </div>
</template>

<style scoped>
.file-upload {
  width: 100%;
}

.upload-area {
  border: 2px dashed #d1d5db;
  border-radius: 12px;
  padding: 40px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
  position: relative;
  overflow: hidden;
}

.upload-area::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(79, 70, 229, 0.1), transparent);
  transition: left 0.6s ease;
}

.upload-area:hover::before {
  left: 100%;
}

.upload-area:hover {
  border-color: #4f46e5;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(79, 70, 229, 0.15);
}

.upload-area.dragging {
  border-color: #4f46e5;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  transform: scale(1.02);
  box-shadow: 0 12px 35px rgba(79, 70, 229, 0.2);
}

.upload-area.uploading {
  opacity: 0.8;
  pointer-events: none;
  background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
}

.upload-icon {
  font-size: 56px;
  margin-bottom: 20px;
  animation: float 3s ease-in-out infinite;
}

.upload-icon.uploading {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.05);
  }
}

.upload-title {
  color: #1f2937;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 12px;
}

.upload-desc {
  color: #6b7280;
  font-size: 14px;
  margin: 0 0 8px 0;
}

.upload-info {
  color: #9ca3af;
  font-size: 13px;
  margin-top: 12px;
  display: block;
}
</style>