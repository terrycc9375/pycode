<script setup>
import { ref, onMounted } from 'vue'
import char from './char.vue'

const isDark = ref(true)
const toggleTheme = () => {
  isDark.value = !isDark.value
  if (isDark.value) {
    document.body.style.backgroundColor = "#202020"
    document.body.style.color = "#f1f1f1"
  }
  else {
    document.body.style.backgroundColor = "#f1f1f1"
    document.body.style.color = "#202020"
  }
  console.log("switched")
}

const applyDarkTheme = () => {
  document.body.style.backgroundColor = "#202020"
  document.body.style.color = "#f1f1f1"
}
onMounted(() => {
  applyDarkTheme()
})

</script>

<template>
  <button @click="toggleTheme" id="toggleButton">
    切換至{{ isDark ? "淺色" : "深色" }}
  </button>
  <char></char>
</template>

<style scoped>
#toggleButton {
  display: flex;
  position: fixed;
  top: 10px;
  right: 10px;
}
</style>
