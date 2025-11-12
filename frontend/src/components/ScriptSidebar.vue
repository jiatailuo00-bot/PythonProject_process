<script setup lang="ts">
import type { ScriptMetadata } from '../types';

const props = defineProps<{
  scripts: ScriptMetadata[];
  selectedId: string;
  filterText: string;
}>();

const emit = defineEmits<{
  (event: 'update:selectedId', value: string): void;
  (event: 'update:filterText', value: string): void;
}>();

const handleFilterChange = (value: string) => {
  emit('update:filterText', value);
};

const handleSelect = (value: string) => {
  emit('update:selectedId', value);
};
</script>

<template>
  <div class="sidebar-wrapper">
    <div class="sidebar-search">
      <el-input
        placeholder="搜索脚本或类别"
        clearable
        :model-value="props.filterText"
        @update:model-value="handleFilterChange"
      />
    </div>

    <el-scrollbar class="sidebar-scroll">
      <el-empty v-if="!props.scripts.length" description="暂无脚本" />
      <el-menu
        v-else
        :default-active="props.selectedId"
        class="sidebar-menu"
        @select="handleSelect"
      >
        <el-menu-item v-for="script in props.scripts" :key="script.id" :index="script.id">
          <div class="menu-item-content">
            <span class="menu-title">{{ script.name }}</span>
            <small>{{ script.category }}</small>
          </div>
        </el-menu-item>
      </el-menu>
    </el-scrollbar>
  </div>
</template>

<style scoped>
.sidebar-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-search {
  padding: 1rem;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-scroll {
  flex: 1;
  height: 0;
  background: linear-gradient(180deg, #e6ecff, #f6f8ff);
  padding: 0.5rem 0;
}

.ruoyi-menu {
  border-right: none;
  background: transparent;
}

.ruoyi-menu .el-menu-item {
  color: rgba(255,255,255,0.8);
  background: transparent;
  border-radius: 8px;
  margin: 0 1rem 0.5rem 1rem;
  transition: all 0.3s ease;
  height: auto;
  line-height: 1.4;
  padding: 0.75rem 1rem;
}

.ruoyi-menu .el-menu-item:hover {
  background: rgba(255,255,255,0.1);
  color: white;
}

.ruoyi-menu .el-menu-item.is-active {
  background: rgba(255,255,255,0.2);
  color: white;
  font-weight: 600;
}

.menu-item-content {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  width: 100%;
}

.menu-title {
  font-weight: 500;
  margin-bottom: 0.25rem;
  color: inherit;
}

menu-title small {
  opacity: 0.7;
  font-size: 0.75rem;
  font-weight: normal;
}

.el-empty {
  padding: 2rem;
  color: rgba(255,255,255,0.6);
}

.el-input {
  --el-input-bg-color: rgba(255,255,255,0.1);
  --el-input-text-color: rgba(255,255,255,0.9);
  --el-input-border-color: rgba(255,255,255,0.2);
  --el-input-focus-border-color: rgba(255,255,255,0.4);
}

.el-input__wrapper {
  background: var(--el-input-bg-color);
  box-shadow: 0 0 0 1px var(--el-input-border-color) inset;
}

.el-input__wrapper:hover {
  box-shadow: 0 0 0 1px var(--el-input-focus-border-color) inset;
}

.el-input__wrapper.is-focus {
  box-shadow: 0 0 0 1px var(--el-input-focus-border-color) inset;
}

.el-input__inner {
  color: var(--el-input-text-color);
  background: transparent;
}

.el-input__inner::placeholder {
  color: rgba(255,255,255,0.5);
}
</style>
