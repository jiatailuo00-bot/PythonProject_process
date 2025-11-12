<script setup lang="ts">
interface HistoryItem {
  id: string;
  scriptName: string;
  timestamp: string;
  success: boolean;
  message: string;
}

const props = defineProps<{ items: HistoryItem[] }>();
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <span class="section-title">最近执行</span>
        <el-tag>{{ props.items.length }} 条</el-tag>
      </div>
    </template>
    <ul class="history-list">
      <li v-if="!props.items.length">
        <span class="meta-badge">暂无记录</span>
      </li>
      <li v-for="item in props.items" :key="item.id">
        <div>
          <strong>{{ item.scriptName }}</strong>
          <p class="meta-badge" style="margin: 2px 0">{{ new Date(item.timestamp).toLocaleString() }}</p>
          <small>{{ item.message }}</small>
        </div>
        <el-tag :type="item.success ? 'success' : 'danger'">
          {{ item.success ? '成功' : '失败' }}
        </el-tag>
      </li>
    </ul>
  </el-card>
</template>
