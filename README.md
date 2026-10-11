# MRP-2: Local Multimodal IoT Orchestration

MRP-2 answers natural-language questions about smart spaces through a locally hosted
multimodal model and on-demand IoT measurements. One `read_sensors` tool combines
camera, temperature, ambient light, and sound observations across rooms. The
prototype monitors spaces; it exposes no actuators.

This repository accompanies **Privacy-Preserving IoT Orchestration with a Local
Multimodal Agent for Smart Space Management**, by Camiel Schmeitz, Che Hang Ng,
Francisco José Delicado González, Grigorios Pachis, Noa Bruninx, Charalampos S.
Kouzinopoulos, and Guangzhi Tang (Maastricht University). See
[CITATION.cff](CITATION.cff). The manuscript reports 92.3% task success across
299 prompts, a 9.1% adversarial violation rate, and median latency of 127 seconds.
These are experiment results, not guarantees for new runs.

## Architecture

```text
Browser / API client
        |
Vue + Vite frontend / nginx :3000
        |
FastAPI gateway :8000
  |-- OpenAI-compatible chat and model-list routes
  |-- LangChain agent / LangGraph workflow
  |      |-- local Qwen model via llama.cpp + ROCm :8080
  |      `-- read_sensors(sensor_names, nodes?)
  |                 |
  |           Mosquitto MQTT :1883
  |                 |
  `----------- STM32 nodes: camera, temperature, light, sound
```

The gateway assembles the request, history, system prompt, tool schema, and known
node identifiers. The model answers, refuses, or selects sensors and nodes. Finding
the warmest empty room requires camera and temperature in one call. Numeric reads
run in parallel; camera frames become base64 image inputs. The same model interprets
the observations and produces an `AgentResponse` with `answer`, `reasoning`,
`sensors_used`, and `confidence`. The gateway returns or streams the result.

One batched tool call per turn is a prompt instruction, not a hard execution policy.
Thinking and non-thinking graphs share process-local conversation checkpoints.
History trimming approximates a token budget of `AGENT_MAX_CONTEXT_MESSAGES * 200`;
`AGENT_MAX_ITERATIONS` controls the graph recursion budget. Restarting loses memory.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/gateway/` | FastAPI app, settings, API schemas/routes, and runtime/provider wiring. |
| `src/agent/app.py` | Agent assembly, streaming, observations, memory, and timing. |
| `src/agent/graph/` | System prompt, graph construction, and history trimming. |
| `src/agent/clients/` | OpenAI-compatible client preserving llama.cpp timing metadata. |
| `src/agent/iot/` | Provider interfaces, stubs, MQTT discovery/request/cache, sensor registry, image conversion, and generated bindings. |
| `src/agent/tools/` | Unified tool, argument schema, numeric output, and camera retrieval. |
| `src/agent/response_schema.py`, `schemas.py` | Structured answers and typed requests, observations, steps, and logs. |
| `src/agent/run_logging.py`, `observability.py` | JSONL logs and optional Langfuse callbacks. |
| `src/agent/demo.py` | Live streamed scenarios and tool-coverage reporting. |
| `src/benchmark.py`, `benchmark_cases.py`, `src/eval/` | Evaluation runner, ground truth, metrics/plots, and optional DeepSeek judge. |
| `frontend/` | Vue UI, styles, nginx proxy, Vite configuration, and npm lockfile. |
| `MRP-IOT/` | STM32CubeIDE project, application firmware, startup/linker files, builder, and bundled ST/ThreadX/NetX Duo/nanopb sources. |
| `iot.proto`, `build-protobuf.bat` | Shared wire schema and regeneration wrapper. |
| `mqtt/` | Broker configurations and Windows launcher. |
| `scripts/` | Simulators, manual pipeline/latency clients, protobuf generator, publication plotting, and camera fixtures. |
| `benchmarks/` | Packet captures/JSON exports, firmware logs, latency notebook, and PDF figures. |
| `eval_results/` | Archived runs, final merged rows/metrics, and publication figures. |
| `tests/` | Offline workflow/tool/API/protocol tests and opt-in live tests. |
| `test_image_room.png`, `data/` | Stub fixture and ignored runtime data directory. |
| `Dockerfile.*`, `docker-compose.yml` | Frontend, gateway, broker, and ROCm deployment. |

Vendor sources and license notices are retained. Framework callbacks, generated
bindings, and drivers remain necessary even without ordinary static call sites.
Local credentials and generated binaries are excluded from Git.

## Docker deployment

The full stack targets **Linux with a ROCm-compatible AMD GPU**. Provide Docker
Compose, a working ROCm-enabled `llama-server` with compatible runtime libraries,
enough model/context memory, and downloaded GGUF weights. The paper used
Qwen3.6-27B on an HP Z2 Mini G1a with an AMD Strix Halo APU and 128 GB unified
memory. Weights and llama.cpp binaries are not bundled.

```bash
cp .env.example .env
# Edit LLAMA_ROCM_DIR, MODELS_DIR, and LLAMA_MODEL_FILE.
docker compose up --build -d
docker compose logs -f llm
```

PowerShell: `Copy-Item .env.example .env`. Put the model in `MODELS_DIR` (default
`./models`) and its matching vision projector at `MODELS_DIR/mmproj-F16.gguf`.
Compose passes `-m` and `--mmproj`; it does **not** download weights. It uses a
36864-token context, six parallel slots, GPU offload, and Q8 KV caches. Adjust
these settings for your hardware/model.

Open `http://localhost:3000`; API docs are at `http://localhost:8000/docs`.
`curl http://localhost:8000/health` checks model reachability. The backend is also
published at `8081`, and MQTT at `1883`. Inside Compose the gateway uses
`llm:8080` and `mqtt:1883`; nginx proxies to `gateway:8000`. Named volumes retain
run logs, broker persistence/logs, and the llama cache.

Default `IOT_PROVIDER=stub` needs no boards: it supplies `stub/node_0`, simulated
temperature, and the bundled camera fixture. Light/sound are unavailable. Use
MQTT hardware or the multi-room simulator for all numeric modalities.

## Local development

Use **Python 3.12+** and **Node.js 24**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # PowerShell: .venv/Scripts/Activate.ps1
pip install -e ".[dev,research]"
BACKEND_URL=http://localhost:8081 MQTT_BROKER_HOST=localhost \
CAMERA_FRAME_FIXTURE_PATH=test_image_room.png uvicorn gateway.main:app --reload
```

In PowerShell set variables first, e.g. `$env:BACKEND_URL="http://localhost:8081"`,
then run `uvicorn gateway.main:app --reload`. Replace Compose hostnames/container
paths in `.env` with local equivalents. `pyproject.toml` declares dependencies and
extras; `requirements.txt` pins the main Docker gateway stack. Pillow/NumPy support
image conversion; research extras provide plotting, statistics, and notebooks.

There is no standalone `agent` console command. Use the API or live demo:

```bash
python -m agent.demo --base-url http://localhost:8000 --timeout 300
cd frontend
npm ci
npm run dev
```

Vite proxies `/health` and `/v1/*` to `localhost:8000`. Optional `frontend/.env`
with `VITE_GATEWAY_BASE_URL` overrides the proxy. `npm run build` type-checks and
builds production assets; `npm run preview` serves them without nginx's API proxy.

## API

| Method and route | Behaviour |
| --- | --- |
| `GET /health` | Backend reachability and healthy/degraded status. |
| `GET /v1/models` | Available models from the backend. |
| `POST /v1/chat/completions` | OpenAI-compatible text/image chat, including streaming. |
| `POST /v1/agent/query` | Complete answer, observations, and optional trace. |
| `POST /v1/agent/stream` | Newline-delimited JSON events for the agent/UI. |

```bash
curl -N http://localhost:8000/v1/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"room-demo","user_input":"Which empty room is warmest?","include_trace":true}'
```

Agent requests accept `thread_id`, `user_input`, `prior_messages`, `include_trace`,
and `enable_thinking`. Reuse thread IDs for memory. Events are `token`,
`reasoning_token`, `trace`, `final`, and `error`. The final event contains
`result.final_answer`, observations, trace, and `run_log`; structured output and
timing/cost summaries are in that log.

Direct chat uses `model: "llm"` and OpenAI-style `messages`; images use base64
`image_url` data URLs. SDK clients requiring a key can use a dummy value for this
unauthenticated endpoint. Explicit sampling parameters override gateway defaults.

## IoT protocol and firmware

STM32 **B-U585I-IOT02A** kits use the **B-CAMS-OMV / OV5640** camera bundle.
ThreadX schedules tasks; NetX Duo/MXCHIP handle Wi-Fi/MQTT. Firmware captures JPEGs
on demand, averages ten temperature readings with 500 ms gaps, derives median
ambient sound from ten RMS/dBFS snapshots, and reads light. No waveform is sent.

| Operation | Request | Response / protobuf type |
| --- | --- | --- |
| Discovery | `node/req` | `node/res` / `Node(id="room/node")` |
| Temperature | `room/node/req/temperature` | `room/node/res/temperature` / `Temperature` |
| Light | `room/node/req/light` | `room/node/res/light` / `Light` |
| Sound | `room/node/req/mic` | `room/node/res/mic` / `Sound` |
| Camera | `room/node/req/camera` | `room/node/res/camera` / `Camera` with JPEG bytes |

Requests have empty payloads; responses encode standalone messages. The registry
accepts `temp`/`temperature` and `mic`/`sound` aliases. Humidity/pressure exist in
the schema but are not tool modalities. Labels are Celsius, lux, and dBFS. The
provider waits for fresh responses, caches by node/sensor, and reports timeouts.
Camera waits up to 45 seconds. `MQTT_NODES` bypasses startup discovery with explicit paths.

Build and connect boards:

1. Copy `MRP-IOT/Core/Inc/network_config.h.example` to `network_config.h` beside it.
   Configure Wi-Fi SSID/password and the broker's numeric IPv4 address. This local
   header is ignored; configured binaries can contain credentials.
2. Import `MRP-IOT` into STM32CubeIDE, or set `ARM_TOOLCHAIN_DIR` to the ARM GNU
   toolchain's `bin` directory and run:

   ```bash
   NODE_NAME=room_a/node_1 python MRP-IOT/build_firmware.py
   ```

   PowerShell: set `$env:NODE_NAME="room_a/node_1"` first. Keep node paths short
   enough for the 32-byte topic buffer including `/req/temperature`. The builder
   produces ELF/BIN/HEX under `MRP-IOT/build/` with the FLASH linker script.
   Program boards with STM32CubeProgrammer.
3. Set `IOT_PROVIDER=mqtt` and configure the gateway broker address. Boards outside
   Docker need the host LAN address, not the Compose service name.

This revision defaults to `CAMERA_R800x480`; the manuscript describes 800×640.
The `CAMERA_HD_ABOVE` raw-720p path is experimental. It also truncates averaged
temperature to integer Celsius and sends sound in hundredths of dBFS despite
Python's dBFS label. Validate units before interpreting new physical measurements;
stored simulator benchmarks use Celsius and unscaled dBFS.

After schema changes, install `protoc` and the nanopb generator, set
`NANOPB_PROTO_DIR` to the directory containing `nanopb.proto`, and run
`python scripts/generate_protobuf.py` (or `build-protobuf.bat`). Put
`protoc-gen-nanopb` on `PATH`; use a generator compatible with the bundled C
runtime (header version 40) and review Python/C bindings together.

## Configuration and privacy

Settings load from environment variables and root `.env`. Examples contain no
working credentials.

| Variables | Purpose |
| --- | --- |
| `LLAMA_ROCM_DIR`, `MODELS_DIR`, `LLAMA_MODEL_FILE` | Compose binary/model mounts. |
| `BACKEND_URL`, `MODEL_NAME`, `REQUEST_TIMEOUT` | Model URL, alias, timeout. |
| `TAILSCALE_HOST`, `GATEWAY_PORT` | Demo/live-client URL; loopback by default, no Tailscale requirement. |
| `IOT_PROVIDER`, `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`, `MQTT_NODES` | Provider, broker, nodes. |
| `MQTT_CONNECT_TIMEOUT`, `MQTT_REQUEST_TIMEOUT` | Connection/numeric-read waits. |
| `CAMERA_FRAME_FIXTURE_PATH`, `SIMULATED_TEMPERATURE_C` | Stub data. |
| `AGENT_MAX_ITERATIONS`, `AGENT_MAX_CONTEXT_MESSAGES`, `AGENT_ENABLE_MEMORY` | Graph limits/history. |
| `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_TOP_K`, `DEFAULT_PRESENCE_PENALTY` | Sampling defaults. |
| `AGENT_RUN_LOG_PATH` | JSONL path; empty disables logging. |
| `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Optional tracing, disabled by default. |
| `LANGFUSE_ENVIRONMENT`, `LANGFUSE_TRACE_NAME`, `LANGFUSE_FLUSH_ON_RUN_END` | Trace labels/flushing. |
| `DEEPSEEK_API_KEY` | Optional external judge. |

Default inference/sensing stay local. Langfuse and DeepSeek send data externally
when enabled. Logs include prompts, answers, and observations, potentially including
base64 camera frames; disable logs or manage retention for real spaces. Images also
enter browser/model context. On-demand acquisition does not provide anonymisation.

Privacy rules are prompt-level guardrails. There is no gateway access control;
MQTT permits anonymous plaintext clients, ports bind to host interfaces, and CORS
is unrestricted. The paper's adversarial failures show refusal instructions are
not an authorization boundary. Production needs authenticated MQTT/TLS and
sensor-access controls independent of the model.

## Reproducing experiments

The source suite has **300** unique cases: 126 single-sensor, 84 multi-sensor,
45 out-of-scope, and 45 adversarial. Published merged results in
`eval_results/results_final/results_final.json` have **299** rows; `adv-emoji` is
absent. Rows retain older `normal`/`multi_tool` categories. Publication plots
recover expected sensors from case definitions to regroup them. Historical metric
JSON keeps its original denominators; 276/299 successful rows yields the paper's
92.3% overall figure. Do not replace archived counts with current suite counts.

Start a local broker/model and run from the repository root:

```bash
BACKEND_URL=http://localhost:8081 MQTT_BROKER_HOST=localhost \
python -m benchmark --no-judge --concurrency 6 --output-dir eval_results/new-run
```

The runner starts the numeric simulator, discovers three `room_ls/C1.*` nodes,
uses `scripts/images/c1.*_frame.jpg` as camera ground truth, and runs independent
non-thinking conversations. It scores answer substrings, tool parameters,
execution/refusal, tokens, latency, and adversarial violations. These are heuristic
checks. `--resume <results.json>` skips completed cases, `--no-plots` skips figures,
and `--quiet` reduces output. The optional judge needs a key and sends responses
externally. Fresh runs need not match archived results.

Regenerate publication figures without inference:

```bash
python scripts/plot_final_results.py --output-dir output/publication-plots
```

Open `benchmarks/benchmark.ipynb` with `benchmarks/` as its working directory for
physical latency analysis. It pairs Wireshark JSON exports and firmware logs,
subtracts acquisition delays, applies per-sensor/settings IQR bounds, and produces
histograms and a stacked communication-time figure. SciencePlots uses LaTeX;
install TeX to render figures. Existing PDF figures remain available.

Manual clients (prefix Python scripts with `python`):

- `scripts/simulate_iot_node.py --host localhost`: adjustable temperature/sound node.
- `scripts/simulate_iot_nodes_eval.py --host localhost`: three-room numeric simulator.
- `scripts/e2e_pipeline_test.py --host localhost`: interactive requests/decoding and image display.
- `scripts/sensors_benchmark.py --host localhost`: original `rls/n1` light-latency collector.
- `mqtt/run-mqtt-server.bat`: Windows Mosquitto test-config launcher.

## Checks and licensing

```bash
python -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
cd frontend && npm ci && npm run build
```

Offline tests use deterministic model doubles for workflows, tools, schemas,
streaming, logs, benchmark integrity, and firmware-compatible MQTT topics. Live
tests require `RUN_LIVE_TESTS=1` and a reachable gateway/model. GitHub Actions
runs offline checks and the frontend build. ARM compilation and physical sensor
verification require the toolchain and boards.

Third-party firmware retains licenses under `MRP-IOT/Drivers/` and
`MRP-IOT/Middlewares/`, plus nanopb notices in `NetXDuo/App/pb*`. Original
project code is distributed under the [MIT license](LICENSE). Model weights have
separate distribution terms.
