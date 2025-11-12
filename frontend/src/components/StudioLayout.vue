<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import type { ScriptMetadata, ScriptRunResponse, UploadedFile } from '../types'
import { fetchScripts, runScript, uploadSingleFile, listUploadedFiles, deleteUploadedFile, deleteUploadedFiles } from '../api'
import ScriptSidebar from './ScriptSidebar.vue'
import ResultPanel from './ResultPanel.vue'
import HistoryPanel from './HistoryPanel.vue'
import FileUpload from './FileUpload.vue'

interface ExecutionRecord {
  id: string
  scriptName: string
  success: boolean
  message: string
  timestamp: string
}

const scripts = ref<ScriptMetadata[]>([])
const selectedScriptId = ref('')
const filterText = ref('')
const formValues = ref<Record<string, unknown>>({ corpus_path: '', output_filename: '' })
const executionResult = ref<ScriptRunResponse | null>(null)
const running = ref(false)
const recentExecutions = ref<ExecutionRecord[]>([])
const uploadedFiles = ref<UploadedFile[]>([])
const uploading = ref(false)
const selectedFilenames = ref<string[]>([])
const batchDeleting = ref(false)

const MAX_HISTORY = 8

const selectedScript = computed(() => scripts.value.find(script => script.id === selectedScriptId.value) || null)
const filteredScripts = computed(() => {
  if (!filterText.value.trim()) return scripts.value
  const term = filterText.value.toLowerCase()
  return scripts.value.filter(script =>
    `${script.name} ${script.description} ${script.category}`.toLowerCase().includes(term)
  )
})

const corpusPath = computed(() => (formValues.value.corpus_path as string) || '')
const outputFileName = computed({
  get: () => (formValues.value.output_filename as string) || '',
  set: (val: string) => {
    formValues.value.output_filename = val
  },
})

const requiresSopResultInput = computed(() => selectedScript.value?.id === 'extract_expected_utterance_parts')

const ensureDefaults = (script: ScriptMetadata) => {
  const defaults: Record<string, unknown> = {}

  // 为每个脚本参数设置默认值
  script.parameters.forEach((param) => {
    if (param.type === 'boolean') {
      defaults[param.name] = false
    } else {
      defaults[param.name] = ''
    }
  })

  // 特殊处理：为了兼容前端UI，保留corpus_path映射到excel_path
  if (script.parameters.some(p => p.name === 'excel_path')) {
    defaults.corpus_path = formValues.value.corpus_path || ''
  } else if (script.parameters.some(p => p.name === 'corpus_path')) {
    defaults.corpus_path = formValues.value.corpus_path || ''
  } else if (script.parameters.some(p => p.name === 'input_file')) {
    defaults.corpus_path = formValues.value.corpus_path || ''
  }

  // 保留其他可能的表单值
  if (formValues.value.output_filename) {
    defaults.output_filename = formValues.value.output_filename
  }

  formValues.value = defaults
}

const loadScripts = async () => {
  try {
    const data = await fetchScripts()
    scripts.value = data
    if (data.length) {
      selectedScriptId.value = data[0].id
      ensureDefaults(data[0])
    }
  } catch (error: any) {
    ElMessage.error(error.message || '无法加载脚本列表')
  }
}

const refreshUploadedFiles = async () => {
  try {
    const response = await listUploadedFiles()
    uploadedFiles.value = response.files
    selectedFilenames.value = []
  } catch (error: any) {
    ElMessage.error(error.message || '获取历史文件失败')
  }
}

const handleFilterChange = (value: string) => {
  filterText.value = value
}

const selectScript = (scriptId: string) => {
  const target = scripts.value.find(script => script.id === scriptId)
  if (target) {
    selectedScriptId.value = target.id
    ensureDefaults(target)
    executionResult.value = null
  }
}

const openDocs = () => {
  window.open('/api/docs', '_blank')
}

const handleFileUploaded = async (result: any) => {
  await refreshUploadedFiles()
  selectUploadedFile(result.path, result.filename || result.path.split('/').pop() || 'uploaded.xlsx')
}

const handleBatchUploaded = async (results: any[]) => {
  await refreshUploadedFiles()
  if (results.length > 0) {
    const firstResult = results[0]
    selectUploadedFile(firstResult.path, firstResult.filename || firstResult.path.split('/').pop() || 'uploaded.xlsx')
  }
}

const selectUploadedFile = (path: string, filename: string) => {
  formValues.value.corpus_path = path
  if (!formValues.value.output_filename) {
    const base = filename.replace(/\.xlsx?$/i, '')
    formValues.value.output_filename = `${base}_result.xlsx`
  }
}

const removeUploadedFile = async (filename: string) => {
  if (!confirm(`删除文件 ${filename} ?`)) return
  try {
    await deleteUploadedFile(filename)
    await refreshUploadedFiles()
    if (corpusPath.value.includes(filename)) {
      formValues.value.corpus_path = ''
    }
    ElMessage.success('删除成功')
  } catch (error: any) {
    ElMessage.error(error.message || '删除失败')
  }
}

const handleSelectionChange = (rows: UploadedFile[]) => {
  selectedFilenames.value = rows.map((row) => row.filename)
}

const handleBatchDelete = async () => {
  if (!selectedFilenames.value.length) {
    ElMessage.warning('请先勾选需要删除的文件')
    return
  }

  if (!confirm(`批量删除 ${selectedFilenames.value.length} 个文件?`)) return

  batchDeleting.value = true
  const targets = [...selectedFilenames.value]
  try {
    const response = await deleteUploadedFiles(targets)
    await refreshUploadedFiles()
    if (targets.some((name) => corpusPath.value.includes(name))) {
      formValues.value.corpus_path = ''
    }
    if (response.failed.length) {
      const errorSummary = response.failed.map((item) => `${item.filename}: ${item.error}`).join('\n')
      ElMessage.error(`部分文件删除失败：\n${errorSummary}`)
    } else {
      ElMessage.success(response.message || '批量删除完成')
    }
  } catch (error: any) {
    ElMessage.error(error.message || '批量删除失败')
  } finally {
    batchDeleting.value = false
  }
}

const handleSubmit = async () => {
  if (!selectedScript.value) {
    ElMessage.warning('请选择脚本')
    return
  }

  if (!corpusPath.value) {
    ElMessage.warning('请上传或选择 Excel 文件')
    return
  }

  if (!outputFileName.value) {
    ElMessage.warning('请填写输出文件名')
    return
  }

  running.value = true
  executionResult.value = null

  try {
    // 准备脚本参数，处理参数名称映射
    const params = { ...formValues.value }
    console.log('原始表单参数:', formValues.value)
    console.log('脚本ID:', selectedScript.value.id)
    console.log('脚本参数:', selectedScript.value.parameters.map(p => p.name))

    // 如果脚本需要excel_path参数，将corpus_path映射到excel_path
    if (selectedScript.value.parameters.some(p => p.name === 'excel_path')) {
      params.excel_path = params.corpus_path
      delete params.corpus_path
      console.log('映射excel_path:', params.excel_path)
    }

    // 如果脚本需要input_file参数，将corpus_path映射到input_file
    if (selectedScript.value.parameters.some(p => p.name === 'input_file')) {
      params.input_file = params.corpus_path
      delete params.corpus_path
      console.log('映射input_file:', params.input_file)
    }

    // corpus_path参数直接使用，不需要映射
    if (selectedScript.value.parameters.some(p => p.name === 'corpus_path')) {
      console.log('使用corpus_path:', params.corpus_path)
    }

    console.log('最终发送参数:', params)
    const result = await runScript(selectedScript.value.id, params)
    executionResult.value = result

    const record: ExecutionRecord = {
      id: Date.now().toString(),
      scriptName: selectedScript.value.name,
      success: result.success,
      message: result.message,
      timestamp: new Date().toISOString(),
    }
    recentExecutions.value.unshift(record)
    recentExecutions.value = recentExecutions.value.slice(0, MAX_HISTORY)

    result.success ? ElMessage.success('脚本执行完成') : ElMessage.error(result.message)
  } catch (error: any) {
    executionResult.value = {
      success: false,
      message: error.message || '执行失败',
      data: {},
    }
    ElMessage.error(error.message || '脚本执行失败')
  } finally {
    running.value = false
  }
}

onMounted(async () => {
  await loadScripts()
  await refreshUploadedFiles()
})
</script>

<template>
  <div class="studio-shell">
    <aside class="studio-sidebar">
      <div class="brand-block">
        <div class="brand-icon">🚀</div>
        <div>
          <strong>Script Studio</strong>
          <p>统一脚本中台</p>
        </div>
      </div>
      <ScriptSidebar
        :scripts="filteredScripts"
        :selected-id="selectedScriptId"
        :filter-text="filterText"
        @update:selected-id="selectScript"
        @update:filter-text="handleFilterChange"
      />
    </aside>

    <main class="studio-main">
      <header class="studio-header">
        <div>
          <h1>脚本控制台</h1>
          <p>集中管理 · 可视化运行 · 一键追踪</p>
        </div>
        <div class="header-badges">
          <el-tag type="success">FastAPI 已启动</el-tag>
          <el-button type="primary" plain @click="openDocs">API 文档</el-button>
        </div>
      </header>

      <section class="workspace" v-if="selectedScript">
        <div class="form-card">
          <div class="form-card__header">
            <div>
              <p class="meta">{{ selectedScript.category }}</p>
              <h2>{{ selectedScript.name }}</h2>
            </div>
            <el-button type="primary" size="large" :loading="running" @click="handleSubmit">
              {{ running ? '执行中...' : '运行脚本' }}
            </el-button>
          </div>
          <p class="meta" v-if="selectedScript.description">{{ selectedScript.description }}</p>

          <div class="upload-surface">
            <div class="upload-section">
              <h3>上传 · 选择 Excel</h3>
              <FileUpload
                accept=".xlsx,.xls"
                :multiple="true"
                :max-size="500 * 1024 * 1024"
                @uploaded="handleFileUploaded"
                @batch="handleBatchUploaded"
              />
              <p class="meta" v-if="corpusPath">已选择：{{ corpusPath }}</p>
              <el-alert
                v-if="requiresSopResultInput"
                type="info"
                :closable="false"
                show-icon
                class="script-hint"
                title="此脚本仅支持使用纯改进版 SOP 标注结果（例如 *_pure_improved.xlsx），请确保输入文件已经完成标注。"
              />
            </div>
            <div class="upload-table-header">
              <h3>历史上传</h3>
              <el-button
                type="danger"
                size="small"
                plain
                :disabled="!selectedFilenames.length || batchDeleting"
                :loading="batchDeleting"
                @click="handleBatchDelete"
              >
                批量删除
              </el-button>
            </div>
            <p class="meta">列表展示后端 uploads 目录中的文件</p>
            <el-table
              :data="uploadedFiles"
              row-key="filename"
              size="small"
              style="margin-top: 12px"
              @selection-change="handleSelectionChange"
            >
              <el-table-column type="selection" width="48" />
              <el-table-column prop="filename" label="文件名" min-width="160" />
              <el-table-column prop="size" label="大小" width="80">
                <template #default="{ row }">
                  {{ (row.size / 1024 / 1024).toFixed(2) }} MB
                </template>
              </el-table-column>
              <el-table-column label="操作" width="180">
                <template #default="{ row }">
                  <el-button type="primary" text @click="selectUploadedFile(row.path, row.filename)">选择</el-button>
                  <el-button type="danger" text @click="removeUploadedFile(row.filename)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>

          <div class="output-section">
            <h3>输出设置</h3>
            <el-input
              v-model="outputFileName"
              placeholder="例如：badcase_result.xlsx"
              clearable
            >
              <template #prepend>输出文件名</template>
            </el-input>
            <p class="meta">系统会将结果保存到后端默认目录，并附加时间戳。</p>
          </div>
        </div>

        <div class="side-panels">
          <ResultPanel :result="executionResult" :loading="running" />
          <HistoryPanel :items="recentExecutions" />
        </div>
      </section>

      <section v-else class="empty-state">
        <p>暂无脚本，请稍后再试</p>
      </section>
    </main>
  </div>
</template>

<style scoped>
.studio-shell {
  display: flex;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f8fafc;
}

.studio-sidebar {
  width: 280px;
  background: #f8fafc;
  color: #1e293b;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 10px rgba(0,0,0,0.05);
  border-right: 1px solid #e2e8f0;
}

.brand-block {
  padding: 1.5rem;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: linear-gradient(135deg, #dfe7ff, #f2f6ff);
}

.brand-icon {
  font-size: 2rem;
  color: #617dff;
}

.brand-block strong {
  font-size: 1.25rem;
  font-weight: 600;
  color: #3f5fd6;
}

.brand-block p {
  margin: 0;
  color: #5f72ab;
  font-size: 0.875rem;
}

.sidebar-hint {
  padding: 1rem 1.5rem;
  margin: 0;
  font-size: 0.8rem;
  color: #64748b;
  text-align: center;
  border-bottom: 1px solid #e2e8f0;
}

.studio-main {
  flex: 1;
  overflow-y: auto;
}

.studio-header {
  background: white;
  padding: 1.5rem 2rem;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.studio-header h1 {
  margin: 0 0 0.25rem 0;
  color: #1e293b;
  font-size: 1.5rem;
  font-weight: 600;
}

.studio-header p {
  margin: 0;
  color: #64748b;
  font-size: 0.875rem;
}

.header-badges {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.workspace {
  padding: 2rem;
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 2rem;
  background: linear-gradient(180deg, #dfe8ff, #f4f7ff);
  border-radius: 16px;
}

.form-card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  border: 1px solid #e2e8f0;
}

.form-card__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.5rem;
}

.form-card__header h2 {
  margin: 0 0 0.25rem 0;
  color: #1e293b;
  font-size: 1.25rem;
  font-weight: 600;
}

.meta {
  color: #64748b;
  font-size: 0.875rem;
  margin: 0;
}

.upload-surface {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.upload-section h3 {
  margin: 0 0 1rem 0;
  color: #1e293b;
  font-size: 1rem;
  font-weight: 600;
}

.script-hint {
  margin-top: 0.75rem;
}

.upload-table-header {
  margin-top: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.upload-table-header h3 {
  margin: 0;
  color: #1e293b;
  font-size: 1rem;
  font-weight: 600;
}

.output-section {
  margin-top: 1.5rem;
}

.output-section h3 {
  margin: 0 0 1rem 0;
  color: #1e293b;
  font-size: 1rem;
  font-weight: 600;
}

.side-panels {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.empty-state {
  padding: 4rem 2rem;
  text-align: center;
  color: #64748b;
}

@media (max-width: 1200px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .side-panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }
}

@media (max-width: 768px) {
  .studio-shell {
    flex-direction: column;
  }

  .studio-sidebar {
    width: 100%;
    height: auto;
    max-height: 300px;
  }

  .studio-header {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }

  .workspace {
    padding: 1rem;
  }

  .side-panels {
    grid-template-columns: 1fr;
  }
}
</style>
