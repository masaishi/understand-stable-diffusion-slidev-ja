<!--Left Narrow Two Cols-->

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps({
  class: {
    type: String,
  },
  layoutClass: {
    type: String,
  },
	leftPercent: {
		type: Number,
		default: 0.5,
	}
})
</script>

<template>
  <div class="slidev-layout w-full h-full gap-4 flex flex-row" :class="props.layoutClass">
    <div class="col-left" :style="{ width: `${props.leftPercent * 100}%` }">
      <slot />
    </div>
    <div class="col-right" :style="{ width: `${(1 - props.leftPercent) * 100}%`}">
      <slot name="right" />
    </div>
  </div>
</template>