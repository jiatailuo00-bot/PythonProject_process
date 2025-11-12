<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus';
import { onMounted, ref, watch } from 'vue';
import { deleteUploadedFile, listUploadedFiles } from '../api';
import type { UploadedFile } from '../types';

const props = withDefaults(defineProps<{
  refreshKey?: number;
}>(), {
  refreshKey: 0,
});

const emit = defineEmits<{
  (event: 'select', file: UploadedFile): void;
}>();

const files = ref<UploadedFile[]>([]);
const loading = ref(false);

const loadFiles = async () => {
  loading.value = true;
  try {
    const { files: list } = await listUploadedFiles();
    files.value = list;
  } catch (error) {
    const message = error instanceof Error ? error.message : '获取文件失败';
    ElMessage.error(message);
  } finally {
    loading.value = false;
  }
};

const copyPath = async (path: string) => {
  try {
    await navigator.clipboard.writeText(path);
    ElMessage.success('已复制路径');
  } catch (error) {
    ElMessage.error('复制失败');
  }
};

const removeFile = async (filename: string) => {
  try {
    await ElMessageBox.confirm(`确认删除 ${filename} 吗？`, '提示', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    });
    await deleteUploadedFile(filename);
    await loadFiles();
    ElMessage.success('已删除');
  } catch (error) {
    if (error === 'cancel') return;
    const message = error instanceof Error ? error.message : '删除失败';
    ElMessage.error(message);
  }
};

onMounted(loadFiles);
watch(
  () => props.refreshKey,
  () => {
    void loadFiles();
  },
);
</script>

<template>
  <el-table
    :data="files"
    v-loading="loading"
    size="small"
    style="width: 100%"
    empty-text="暂无文件"
  >
    <el-table-column label="文件名">
      <template #default="{ row }">
        <div style="font-weight: 500">{{ row.filename }}</div>
        <small class="meta-badge">{{ row.path }}</small>
      </template>
    </el-table-column>
    <el-table-column label="大小" width="120">
      <template #default="{ row }">
        {{ (row.size / 1024 / 1024).toFixed(2) }} MB
      </template>
    </el-table-column>
    <el-table-column label="上传时间" width="180">
      <template #default="{ row }">
        {{ new Date(row.created_time).toLocaleString() }}
      </template>
    </el-table-column>
    <el-table-column label="操作" width="200" align="center">
      <template #default="{ row }">
        <el-button size="small" type="primary" @click="emit('select', row)">选择</el-button>
        <el-button size="small" @click="copyPath(row.path)">复制</el-button>
        <el-button size="small" type="danger" @click="removeFile(row.filename)">删除</el-button>
      </template>
    </el-table-column>
  </el-table>
</template>
