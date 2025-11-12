<script setup lang="ts">
import type { ScriptRunResponse } from '../types';

const props = defineProps<{ result: ScriptRunResponse | null; loading: boolean }>();
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <span class="section-title">运行结果</span>
        <el-tag v-if="props.result" :type="props.result.success ? 'success' : 'danger'">
          {{ props.result.success ? 'SUCCESS' : 'ERROR' }}
        </el-tag>
      </div>
    </template>

    <div v-if="loading" style="padding: 24px 0">
      <el-skeleton :rows="3" animated />
    </div>

    <div v-else-if="!props.result" style="text-align: center; padding: 24px 0">
      <p class="meta-badge">选择脚本并填写参数后即可运行</p>
    </div>

    <div v-else>
      <p>{{ props.result.message }}</p>
      <div v-if="Object.keys(props.result.data || {}).length" style="margin-top: 12px">
        <h4>返回数据</h4>
        <pre class="result-pre">{{ JSON.stringify(props.result.data, null, 2) }}</pre>
      </div>
      <div v-if="props.result.logs" style="margin-top: 12px">
        <h4>日志输出</h4>
        <pre class="result-pre">{{ props.result.logs }}</pre>
      </div>
    </div>
  </el-card>
</template>
