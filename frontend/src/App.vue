<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { marked } from 'marked'

const DEFAULT_GATEWAY_BASE_URL = ''
const GATEWAY_BASE_URL =
  import.meta.env.VITE_GATEWAY_BASE_URL?.replace(/\/$/, '') || DEFAULT_GATEWAY_BASE_URL
const AGENT_STREAM_URL = `${GATEWAY_BASE_URL}/v1/agent/stream`

type RunState = 'idle' | 'running' | 'done' | 'error'
type AgentEventType = 'token' | 'reasoning_token' | 'trace' | 'final' | 'error'
type ToolCallState = 'pending' | 'executing' | 'done'

type AgentEvent = {
  event?: AgentEventType
  stage?: string
  payload?: Record<string, unknown>
  result?: AgentResult
  detail?: Record<string, unknown>
}

type AgentResult = {
  thread_id: string
  final_answer: string
  tool_observations: ToolObservation[]
  run_log?: RunLog | null
}

type ToolObservation = {
  tool_name: string
  input_summary: string
  output: any
}

type LiveToolCall = {
  id: string
  tool: string
  name: string
  state: ToolCallState
  durationMs?: number
  cards: ToolCard[]
}

type ToolCard = {
  tool: string
  kind: 'sensor' | 'camera' | 'node_list' | 'unknown'
  title: string
  node?: string
  value?: string
  unit?: string
  imageUrl?: string
  text?: string
}

type RunLog = {
  run_id?: string
  thread_id: string
  total_duration_ms?: number
  steps?: StepRecord[]
}

type StepRecord = {
  sequence: number
  stage: string
  name: string
  duration_ms?: number
  summary?: string
  usage?: Record<string, number>
}

type LogEntry = {
  id: string
  level: 'info' | 'tool' | 'model' | 'error' | 'final'
  title: string
  detail: string
  elapsedMs?: number
  ts: number
}

marked.setOptions({ breaks: true, gfm: true })

const prompt = ref('')
const threadId = ref(`ui-${crypto.randomUUID().slice(0, 8)}`)
const runState = ref<RunState>('idle')
const streamedAnswer = ref('')
const finalAnswer = ref('')
const userPrompt = ref('')         // displayed prompt bubble
const lastUserPrompt = ref('')     // saved for retry
const logs = ref<LogEntry[]>([])
const toolObservations = ref<ToolObservation[]>([])
const totalDurationMs = ref<number | null>(null)
const errorMessage = ref('')
const logList = ref<HTMLElement | null>(null)
const answerScroll = ref<HTMLElement | null>(null)
const logPanelOpen = ref(true)
const sidebarTab = ref<'log' | 'images'>('log')
const lightboxImage = ref<string | null>(null)
const elapsedMs = ref(0)
const runStartTs = ref<number | null>(null)
const traceExpanded = ref(false)   // fold-out for thinking trace
const streamedReasoning = ref('') // thinking trace text
const enableThinking = ref(true)  // thinking mode toggle
const conversationStarted = ref(false) // locks toggle after first message

const liveToolCalls = ref<LiveToolCall[]>([])
let toolCallSeq = 0
let activeRun: AbortController | null = null
let elapsedTimer: ReturnType<typeof setInterval> | null = null

const canSend = computed(() => prompt.value.trim().length > 0 && runState.value !== 'running')
const answerText = computed(() => finalAnswer.value || streamedAnswer.value)
const renderedAnswer = computed(() => {
  if (!answerText.value) return ''
  return marked.parse(answerText.value) as string
})
const showConversation = computed(
  () => !!userPrompt.value || !!answerText.value || allToolGroups.value.length > 0 || runState.value === 'running'
)

const allToolGroups = computed<LiveToolCall[]>(() => {
  const completed: LiveToolCall[] = toolObservations.value.map((obs, i) => ({
    id: `done-${i}`,
    tool: obs.tool_name,
    name: formatToolName(obs.tool_name),
    state: 'done' as ToolCallState,
    cards: buildToolCards(obs),
  }))
  return [...liveToolCalls.value, ...completed]
})

const displayElapsed = computed(() => {
  if (totalDurationMs.value !== null) return formatDuration(totalDurationMs.value)
  if (runState.value === 'running') return formatDuration(elapsedMs.value)
  return null
})

// True while agent is still thinking (no answer yet)
const isThinking = computed(() => runState.value === 'running' && !answerText.value)

// True once done and we have tool groups to show in summary
const hasTrace = computed(() => allToolGroups.value.length > 0)

// Extract camera frames from all tool observations for the images tab
const cameraFrames = computed(() => {
  const frames: { node: string; title: string; imageUrl: string }[] = []
  for (const obs of toolObservations.value) {
    if (obs.tool_name !== 'read_sensors' || !Array.isArray(obs.output)) continue
    const labels = obs.output.filter(
      (b: any) => b?.type === 'text' && b.text?.startsWith('Camera frame from')
    )
    const images = obs.output.filter((b: any) => b?.type === 'image_url')
    for (let i = 0; i < images.length; i++) {
      const url = images[i]?.image_url?.url
      if (!url) continue
      const node = labels[i]?.text?.replace('Camera frame from ', '') || 'Unknown node'
      frames.push({ node, title: `Camera — ${node}`, imageUrl: url })
    }
  }
  return frames
})

function openLightbox(url: string) { lightboxImage.value = url }
function closeLightbox() { lightboxImage.value = null }

// Close lightbox on Escape
watch(lightboxImage, (open) => {
  if (!open) return
  const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeLightbox() }
  document.addEventListener('keydown', onKey)
  watch(lightboxImage, () => document.removeEventListener('keydown', onKey), { once: true })
})

const SENSOR_LABELS: Record<string, { label: string; unit?: string }> = {
  temperature: { label: 'Temperature', unit: '°C' },
  temp: { label: 'Temperature', unit: '°C' },
  light: { label: 'Light', unit: 'lux' },
  sound: { label: 'Sound', unit: 'dBFS' },
  camera: { label: 'Camera' },
}

function formatToolName(name: string): string {
  const map: Record<string, string> = {
    read_sensors: 'Reading sensors',
  }
  return map[name] || name.replace(/_/g, ' ')
}

function buildToolCards(obs: ToolObservation): ToolCard[] {
  const output = obs.output
  const tool = obs.tool_name

  if (tool === 'read_sensors') {
    const cards: ToolCard[] = []

    // When camera was requested, the output is an array of content blocks.
    // The first block is always a text block with JSON readings.
    if (Array.isArray(output)) {
      // Parse JSON readings from the first text block.
      const textBlock = output.find((b: any) => b?.type === 'text')
      if (textBlock?.text) {
        try {
          const parsed = JSON.parse(textBlock.text)
          if (parsed.readings) {
            const readings = parsed.readings as Record<string, Record<string, { value: number; unit?: string }>>
            const nodeOrder = (parsed.nodes_queried as string[]) || Object.keys(readings)
            for (const node of nodeOrder) {
              const nodeData = readings[node]
              if (!nodeData) continue
              for (const [sensor, data] of Object.entries(nodeData)) {
                const spec = SENSOR_LABELS[sensor]
                cards.push({
                  tool, kind: 'sensor',
                  title: spec?.label || sensor,
                  node,
                  value: data?.value !== undefined ? String(data.value) : undefined,
                  unit: data?.unit || spec?.unit || undefined,
                })
              }
            }
          }
        } catch { /* not JSON — skip */ }
      }
      // Extract camera frames from image blocks.
      const cameraLabels = output
        .filter((b: any) => b?.type === 'text' && b.text?.startsWith('Camera frame from'))
      const imageBlocks = output.filter((b: any) => b?.type === 'image_url')
      for (let i = 0; i < imageBlocks.length; i++) {
        cards.push({
          tool, kind: 'camera',
          title: cameraLabels[i]?.text || 'Camera Frame',
          imageUrl: imageBlocks[i]?.image_url?.url,
        })
      }
      return cards.length ? cards : [{ tool, kind: 'unknown', title: 'Sensor Read', text: JSON.stringify(output) }]
    }

    // Pure numeric query — output is a parsed JSON object with a .readings key.
    if (output && typeof output === 'object' && !Array.isArray(output) && output.readings) {
      const readings = output.readings as Record<string, Record<string, { value: number; unit?: string }>>
      const nodeOrder = (output.nodes_queried as string[]) || Object.keys(readings)
      for (const node of nodeOrder) {
        const nodeData = readings[node]
        if (!nodeData) continue
        for (const [sensor, data] of Object.entries(nodeData)) {
          const spec = SENSOR_LABELS[sensor]
          cards.push({
            tool, kind: 'sensor',
            title: spec?.label || sensor,
            node,
            value: data?.value !== undefined ? String(data.value) : undefined,
            unit: data?.unit || spec?.unit || undefined,
          })
        }
      }
      return cards
    }
    return [{ tool, kind: 'unknown', title: 'Sensor Read', text: JSON.stringify(output) }]
  }

  return [{ tool, kind: 'unknown', title: tool, text: String(output || '') }]
}

function formatDuration(ms?: number | null): string {
  if (!ms) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function summarizePayload(payload: Record<string, unknown>): string {
  const summary = payload.summary
  if (typeof summary === 'string' && summary.trim()) return summary
  const usage = payload.usage
  if (usage && typeof usage === 'object') {
    return Object.entries(usage as Record<string, unknown>).map(([k, v]) => `${k}: ${v}`).join(', ')
  }
  return ''
}

function resetRunState() {
  activeRun?.abort()
  activeRun = null
  runState.value = 'idle'
  streamedAnswer.value = ''
  finalAnswer.value = ''
  logs.value = []
  toolObservations.value = []
  totalDurationMs.value = null
  errorMessage.value = ''
  liveToolCalls.value = []
  streamedReasoning.value = ''
  toolCallSeq = 0
  elapsedMs.value = 0
  runStartTs.value = null
  traceExpanded.value = false
  lightboxImage.value = null
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
}

function addLog(entry: Omit<LogEntry, 'id' | 'ts'>) {
  logs.value.push({ id: `${Date.now()}-${logs.value.length}`, ts: Date.now(), ...entry })
  nextTick(() => logList.value?.scrollTo({ top: logList.value.scrollHeight, behavior: 'smooth' }))
}

function handleTrace(event: AgentEvent) {
  const payload = event.payload || {}
  const stage = String(event.stage || payload.stage || 'step')
  const name = String(payload.name || stage)
  const duration = Number(payload.duration_ms || 0)
  const level: LogEntry['level'] = stage === 'tools' ? 'tool' : stage === 'model' ? 'model' : 'info'

  addLog({ level, title: name, detail: summarizePayload(payload), elapsedMs: duration })

  if (stage === 'model' && name.startsWith('tool_call:')) {
    const toolName = name.replace('tool_call:', '').trim()
    liveToolCalls.value.push({
      id: `live-${++toolCallSeq}`, tool: toolName,
      name: formatToolName(toolName), state: 'pending', cards: [],
    })
  }

  if (stage === 'tools' && name) {
    const toolNames = name.split(',').map((s: string) => s.trim())
    for (const tn of toolNames) {
      const match = liveToolCalls.value.find(tc => tc.tool === tn && tc.state === 'pending')
      if (match) { match.state = 'executing'; match.durationMs = duration }
      else {
        liveToolCalls.value.push({
          id: `live-${++toolCallSeq}`, tool: tn,
          name: formatToolName(tn), state: 'executing', durationMs: duration, cards: [],
        })
      }
    }
  }
}

function handleFinal(result: AgentResult) {
  finalAnswer.value = result.final_answer || streamedAnswer.value
  toolObservations.value = result.tool_observations || []
  totalDurationMs.value = result.run_log?.total_duration_ms || null
  runState.value = 'done'
  liveToolCalls.value = []
  traceExpanded.value = false   // collapse on completion
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
  addLog({
    level: 'final', title: 'Completed',
    detail: totalDurationMs.value ? formatDuration(totalDurationMs.value) : '',
    elapsedMs: totalDurationMs.value || undefined,
  })
}

function handleAgentEvent(event: AgentEvent) {
  if (event.event === 'reasoning_token') {
    const text = event.payload?.text
    if (typeof text === 'string') {
      streamedReasoning.value += text
    }
    return
  }
  if (event.event === 'token') {
    const text = event.payload?.text
    if (typeof text === 'string') {
      streamedAnswer.value += text
      nextTick(() => answerScroll.value?.scrollTo({ top: answerScroll.value.scrollHeight }))
    }
    return
  }
  if (event.event === 'trace') { handleTrace(event); return }
  if (event.event === 'final' && event.result) { handleFinal(event.result); return }
  if (event.event === 'error') {
    const message = String(event.detail?.message || 'Agent stream failed')
    errorMessage.value = message
    runState.value = 'error'
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
    addLog({ level: 'error', title: 'Stream error', detail: message })
  }
}

async function runAgentWith(input: string) {
  if (!input || runState.value === 'running') return

  resetRunState()
  userPrompt.value = input
  lastUserPrompt.value = input
  runState.value = 'running'
  activeRun = new AbortController()
  runStartTs.value = Date.now()
  elapsedTimer = setInterval(() => {
    if (runStartTs.value) elapsedMs.value = Date.now() - runStartTs.value
  }, 100)

  addLog({ level: 'info', title: 'Request sent', detail: input })
  conversationStarted.value = true

  try {
    const response = await fetch(AGENT_STREAM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        thread_id: threadId.value,
        user_input: input,
        include_trace: true,
        enable_thinking: enableThinking.value,
      }),
      signal: activeRun.signal,
    })

    if (!response.ok || !response.body) throw new Error(`Gateway returned ${response.status}`)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.trim()) continue
        handleAgentEvent(JSON.parse(line) as AgentEvent)
      }
    }

    if (buffer.trim()) handleAgentEvent(JSON.parse(buffer) as AgentEvent)
    if (runState.value === 'running') runState.value = 'done'
  } catch (error) {
    if ((error as Error).name === 'AbortError') return
    const message = error instanceof Error ? error.message : 'Unknown stream failure'
    errorMessage.value = message
    runState.value = 'error'
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
    addLog({ level: 'error', title: 'Request failed', detail: message })
  } finally {
    activeRun = null
  }
}

function runAgent() {
  const input = prompt.value.trim()
  if (!input) return
  prompt.value = ''
  runAgentWith(input)
}

function retryAgent() {
  if (lastUserPrompt.value) runAgentWith(lastUserPrompt.value)
}

function stopAgent() {
  activeRun?.abort()
  activeRun = null
  runState.value = 'idle'
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
  addLog({ level: 'info', title: 'Run stopped', detail: 'Stream cancelled by user.' })
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    runAgent()
  }
}

const QUICK_TILES = [
  {
    title: 'Fire emergency',
    icon: '🔥',
    prompt: 'Emergency! There may be a fire in the building. Check every room immediately: use the camera to look for people who need rescue, read temperature and light levels to find the fire source, and use sound/mic to listen for fire alarms or signs of danger. Give a room-by-room report for firefighters — which rooms are clear, which have people, and where the fire seems to be.',
  },
  {
    title: 'Security sweep',
    icon: '🛡️',
    prompt: 'Run a security sweep of the building. Check every room: use the camera to verify whether each room is empty or occupied, and use the sound/mic sensor to detect any unusual noise. Report any rooms that are occupied or have suspicious sound levels.',
  },
  {
    title: 'Average temperature',
    icon: '🌡️',
    prompt: 'Read the temperature from every room and calculate the average temperature across the entire building. Report each room\'s temperature and the overall average.',
  },
  {
    title: 'Cleaning check',
    icon: '🧹',
    prompt: 'The cleaning team needs a report. Check every room using the camera to see which rooms are dirty, messy, or need cleaning attention. Use light levels if helpful. Report which rooms need cleaning and which are already clean.',
  },
]

function runQuickTile(presetPrompt: string) {
  runAgentWith(presetPrompt)
}

onUnmounted(() => {
  activeRun?.abort()
  if (elapsedTimer) clearInterval(elapsedTimer)
})
</script>

<template>
  <div id="app">
    <main id="workspace" :class="{ 'log-open': logPanelOpen }">

      <!-- Conversation column -->
      <section id="conversation">

        <!-- Scrollable content area -->
        <div ref="answerScroll" class="answer-area">
          <div class="content-wrap">

            <!-- Welcome -->
            <Transition name="fade" mode="out-in">
              <div v-if="!showConversation" key="welcome" class="welcome">
                <p class="welcome-message">What would you like to know about the rooms?</p>
                <div class="quick-tiles">
                  <button
                    v-for="tile in QUICK_TILES"
                    :key="tile.title"
                    class="quick-tile"
                    @click="runQuickTile(tile.prompt)"
                  >
                    <span class="quick-tile-icon">{{ tile.icon }}</span>
                    <span class="quick-tile-title">{{ tile.title }}</span>
                  </button>
                </div>
              </div>

              <!-- Active conversation -->
              <div v-else key="convo" class="conversation">

                <!-- User prompt bubble -->
                <div v-if="userPrompt" class="user-message">
                  <p>{{ userPrompt }}</p>
                </div>

                <!-- Thinking trace — expands/collapses -->
                <div v-if="hasTrace || isThinking" class="trace-block">
                  <!-- While thinking: animated label, no toggle -->
                  <div v-if="isThinking" class="trace-summary thinking">
                    <span class="thinking-dots">
                      <span></span><span></span><span></span>
                    </span>
                    <span class="trace-summary-text">Thinking…</span>
                  </div>

                  <!-- After done: collapsible toggle -->
                  <button
                    v-else-if="hasTrace"
                    class="trace-toggle"
                    @click="traceExpanded = !traceExpanded"
                  >
                    <svg
                      class="trace-chevron"
                      :class="{ open: traceExpanded }"
                      width="12" height="12" viewBox="0 0 12 12" fill="none"
                    >
                      <path d="M3 4.5l3 3 3-3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <span class="trace-summary-text">
                      Used {{ allToolGroups.length }} tool{{ allToolGroups.length !== 1 ? 's' : '' }}
                    </span>
                    <span v-if="displayElapsed" class="trace-duration">{{ displayElapsed }}</span>
                  </button>

                  <!-- Trace detail — shown while thinking OR when expanded -->
                  <div v-show="isThinking || traceExpanded" class="trace-detail">
                    <!-- Thinking/reasoning stream -->
                    <div v-if="streamedReasoning" class="reasoning-stream">{{ streamedReasoning }}</div>
                    <TransitionGroup name="tool-slide" tag="div" class="tool-list">
                      <div
                        v-for="group in allToolGroups"
                        :key="group.id"
                        class="tool-item"
                        :data-state="group.state"
                      >
                        <div class="tool-item-row">
                          <!-- State icon -->
                          <span class="tool-state-icon" aria-hidden="true">
                            <span v-if="group.state === 'pending'" class="tool-spinner"></span>
                            <span v-else-if="group.state === 'executing'" class="tool-pulse"></span>
                            <svg v-else width="10" height="10" viewBox="0 0 10 10" fill="none">
                              <path d="M2 5l2.5 2.5 3.5-3.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                          </span>
                          <span class="tool-name">{{ group.name }}</span>
                          <span v-if="group.durationMs" class="tool-duration">{{ formatDuration(group.durationMs) }}</span>
                        </div>

                        <!-- Sensor/camera results inline -->
                        <div v-if="group.cards.length && group.state === 'done'" class="tool-results">
                          <template v-for="(card, ci) in group.cards" :key="`${group.id}-${ci}`">

                            <!-- Sensor: inline text row -->
                            <div v-if="card.kind === 'sensor'" class="result-row">
                              <span class="result-label">{{ card.title }}</span>
                              <span v-if="card.node" class="result-node">{{ card.node }}</span>
                              <span class="result-value">{{ card.value ?? '—' }}<span v-if="card.unit" class="result-unit"> {{ card.unit }}</span></span>
                            </div>

                            <!-- Camera -->
                            <div v-else-if="card.kind === 'camera'" class="result-camera">
                              <span class="result-label">{{ card.title }}</span>
                              <img v-if="card.imageUrl" :src="card.imageUrl" alt="Camera frame" class="result-image" />
                              <span v-else class="result-faint">No image data.</span>
                            </div>

                            <!-- Unknown / fallback -->
                            <div v-else class="result-row">
                              <span class="result-label">{{ card.title }}</span>
                              <span class="result-text">{{ card.text }}</span>
                            </div>

                          </template>
                        </div>
                      </div>
                    </TransitionGroup>
                  </div>
                </div>

                <!-- Agent answer -->
                <Transition name="answer-fade">
                  <div v-if="answerText" class="agent-answer">
                    <div class="answer-markdown" v-html="renderedAnswer"></div>
                    <span v-if="runState === 'running'" class="typing-cursor" aria-hidden="true"></span>

                    <!-- Retry button — shown after completion -->
                    <div v-if="runState === 'done' || runState === 'error'" class="answer-actions">
                      <button class="btn-retry" @click="retryAgent" title="Regenerate response">
                        <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                          <path d="M11 6.5A4.5 4.5 0 1 1 6.5 2H9M9 2l-2 2M9 2l-2-2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Retry
                      </button>
                    </div>
                  </div>
                </Transition>

                <!-- Error -->
                <Transition name="answer-fade">
                  <div v-if="errorMessage" class="error-notice">
                    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <circle cx="6.5" cy="6.5" r="5.5" stroke="currentColor" stroke-width="1.2" fill="none"/>
                      <path d="M6.5 4v3M6.5 9h.01" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
                    </svg>
                    {{ errorMessage }}
                    <button class="btn-retry inline" @click="retryAgent">Retry</button>
                  </div>
                </Transition>

              </div>
            </Transition>

          </div>
        </div>

        <!-- Compose island — constrained width, centred -->
        <div class="compose-wrap">
          <div class="compose-island">
            <textarea
              v-model="prompt"
              placeholder="Ask about the room, sensors, or current conditions…"
              rows="3"
              spellcheck="false"
              :disabled="runState === 'running'"
              @keydown="onKeydown"
            ></textarea>
            <div class="compose-footer">
              <div class="compose-left">
                <span class="compose-hint">Enter to send · Shift+Enter for new line</span>
              </div>
              <div class="compose-actions">
                <!-- Thinking toggle — locked after first message -->
                <label
                  class="think-toggle"
                  :class="{ locked: conversationStarted }"
                  :title="conversationStarted ? 'Thinking mode is locked for this conversation' : enableThinking ? 'Thinking mode on — click to disable' : 'Thinking mode off — click to enable'"
                >
                  <span class="think-label">Think</span>
                  <input
                    type="checkbox"
                    :checked="enableThinking"
                    :disabled="conversationStarted"
                    @change="enableThinking = ($event.target as HTMLInputElement).checked"
                  />
                  <span class="think-slider" aria-hidden="true"></span>
                </label>
                <Transition name="btn-fade">
                  <button v-if="runState === 'running'" class="btn-stop" @click="stopAgent" type="button">
                    <span class="stop-sq" aria-hidden="true"></span>
                    Stop
                  </button>
                </Transition>
                <button class="btn-send" :disabled="!canSend" @click="runAgent" type="button">
                  <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                    <path d="M11.5 6.5L1.5 2l2 4.5-2 4.5 10-4.5z" fill="currentColor"/>
                  </svg>
                  Send
                </button>
              </div>
            </div>
          </div>

          <!-- Log toggle sits below compose, right-aligned -->
          <div class="below-compose">
            <button
              class="log-toggle"
              :class="{ active: logPanelOpen }"
              @click="logPanelOpen = !logPanelOpen"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <rect x="1" y="1.5" width="10" height="1.3" rx="0.65" fill="currentColor"/>
                <rect x="1" y="5.35" width="7" height="1.3" rx="0.65" fill="currentColor"/>
                <rect x="1" y="9.2" width="8.5" height="1.3" rx="0.65" fill="currentColor"/>
              </svg>
              {{ logPanelOpen ? 'Hide log' : 'Run log' }}
            </button>
          </div>
        </div>

      </section>

      <!-- Log / Images sidebar -->
      <aside v-show="logPanelOpen" id="log-panel">
        <!-- Tab bar -->
        <div class="sidebar-tabs">
          <button
            class="sidebar-tab"
            :class="{ active: sidebarTab === 'log' }"
            @click="sidebarTab = 'log'"
          >Run log</button>
          <button
            class="sidebar-tab"
            :class="{ active: sidebarTab === 'images' }"
            @click="sidebarTab = 'images'"
          >
            Images
            <span v-if="cameraFrames.length" class="tab-badge">{{ cameraFrames.length }}</span>
          </button>
          <Transition name="fade">
            <span v-if="totalDurationMs" class="log-duration">{{ formatDuration(totalDurationMs) }}</span>
          </Transition>
        </div>

        <!-- Log view -->
        <div v-show="sidebarTab === 'log'" ref="logList" class="log-entries">
          <div v-if="logs.length === 0" class="log-empty">No events yet</div>

          <TransitionGroup v-else name="log-item" tag="div" class="log-stack">
            <div
              v-for="entry in logs"
              :key="entry.id"
              class="log-entry"
              :data-level="entry.level"
            >
              <div class="log-indicator" aria-hidden="true"></div>
              <div class="log-content">
                <div class="log-row">
                  <span class="log-title">{{ entry.title }}</span>
                  <span v-if="entry.elapsedMs" class="log-time">{{ formatDuration(entry.elapsedMs) }}</span>
                </div>
                <p v-if="entry.detail" class="log-detail">{{ entry.detail }}</p>
              </div>
            </div>
          </TransitionGroup>
        </div>

        <!-- Images view -->
        <div v-show="sidebarTab === 'images'" class="images-panel">
          <div v-if="cameraFrames.length === 0" class="log-empty">
            <p>No camera frames yet.</p>
            <p class="images-hint">Ask the agent to check a room with the camera to see images here.</p>
          </div>
          <div v-else class="images-grid">
            <button
              v-for="(frame, fi) in cameraFrames"
              :key="fi"
              class="image-thumb"
              @click="openLightbox(frame.imageUrl)"
            >
              <img :src="frame.imageUrl" :alt="frame.title" />
              <span class="image-thumb-label">{{ frame.node }}</span>
            </button>
          </div>
        </div>
      </aside>

      <!-- Lightbox overlay -->
      <Teleport to="body">
        <Transition name="lightbox-fade">
          <div
            v-if="lightboxImage"
            class="lightbox-overlay"
            @click.self="closeLightbox"
          >
            <button class="lightbox-close" @click="closeLightbox" aria-label="Close image">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
              </svg>
            </button>
            <img :src="lightboxImage" class="lightbox-image" alt="Camera frame full size" />
          </div>
        </Transition>
      </Teleport>

    </main>
  </div>
</template>