# BAVA 启动与运行手册

本文是 BAVA（Budget-Aware Visual Allocator）系统的运行手册——冷启动、跑通单流/多流、看指标、排错。
本文在 **两个位置保持同步**，随便 SSH 到哪都能看到：

- 边缘（本机）: `/home/admin123/tangxuan/edge/trace_doc.md`
- 云端（210.45.123.163）: `/home/mambauser/tangxuan/online_vllm/intake/trace_doc.md`

修改一侧后请记得同步到另一侧（scp 即可）。

> 系统设计细节与每一阶段踩过的坑见 `debug_report.md`（只放在边缘侧 `/home/admin123/tangxuan/debug_report.md`）。本文只讲「怎么启动、怎么跑、怎么看」。

---

## 1. 全局拓扑

```
  边缘（本机 WSL2 /home/admin123/tangxuan/edge）
        │
        │  WebSocket (text 控制 / binary H.264 包)
        │  经 SSH 隧道 :19100 → 云端 :9100
        ▼
  云端 intake（FastAPI+websockets，/home/mambauser/tangxuan/online_vllm/intake）
        │
        │  HTTP (loopback :8001)
        ▼
  云端 vLLM（OpenAI API server, FP8 Qwen3-VL-8B, --enable-online-prefill）
```

三侧的角色：

- **边缘**：按 ρ 挑 H.264 包（PacketGame 风格，不解码像素），打包成 WebSocket 帧上行；同一条 WS 下行接收云端的 ρ/α 更新。
- **intake**：收包→按 window_id 聚齐→PyAV 软解→α 缩放→JPEG→HTTP 喂 vLLM online_prefill；后台 `BavaController` 每 500ms 拉 vLLM `/metrics` 跑 M5 离散更新。
- **vLLM**：标准实例（不改源码），暴露 `/v1/online_prefill/*` 三件套 + `/metrics`。

---

## 2. 前置条件

### 2.1 云端（210.45.123.163）

| 项目 | 位置 |
|---|---|
| SSH | `ssh -p 2222 -i ~/.ssh/jupyterhub.pem mambauser@210.45.123.163`（本机 ssh key 权限必须 600）|
| vLLM 源码 | `/home/mambauser/tangxuan/online_vllm/vllm/` — 不要动 |
| vLLM venv | `/home/mambauser/tangxuan/online_vllm/.venv/` (uv-managed；用 `uv pip install ...` 加依赖)|
| intake 代码 | `/home/mambauser/tangxuan/online_vllm/intake/`（和 vllm 平级）|
| 模型 | `/home/mambauser/tangxuan/models/Qwen3-VL-8B-Instruct-FP8`（也有非 FP8 的，以 8001 端口为准）|
| 数据集 | `/home/mambauser/tangxuan/ucf_crime_hf/ucf-crime-dataset.zip`（11.6GB）|
| UCF 转码产物 | `/home/mambauser/tangxuan/ucf_crime_hf/staged/*.mp4` |

端口约定：

| 端口 | 用途 | 外部可达？ |
|---|---|---|
| 8000 | 历史上的非 FP8 vLLM（现在可能没起）| 看情况 |
| 8001 | **现用的 FP8 vLLM**（tp=1 CUDA:2, max_model_len=32768）| loopback only |
| 8002 | bf16 非 FP8，max_model_len=2048，CUDA:3 | loopback only |
| 8003 | FP8，max_model_len=2048，CUDA:4 | loopback only |
| 9100 | **intake WebSocket+HTTP**（本系统主入口）| 公网被 Node Exporter 占了！走 SSH 隧道 |

云端 python 依赖（`.venv/` 内已装）：`fastapi websockets uvicorn httpx cv2 av numpy`。

### 2.2 边缘（本机 WSL2）

| 项目 | 位置 |
|---|---|
| edge 代码 | `/home/admin123/tangxuan/edge/` |
| SSH 私钥 | `~/.ssh/jupyterhub.pem` (chmod 600) |
| Python | 系统 Python 3.12，`--break-system-packages` 装 `websockets>=12`|
| 示例数据 | `edge/data/test.mp4`（合成 5s 测试片）、 `edge/data/ucf/*.mp4`（10 个 UCF 样本）|

```bash
# 一次性装依赖
python3 -m pip install --user --break-system-packages -r /home/admin123/tangxuan/edge/requirements.txt
```

---

## 3. 冷启动流程

### 3.1 云端：vLLM（如果没起）

先看 :8001 活没活：

```bash
ssh -p 2222 -i ~/.ssh/jupyterhub.pem mambauser@210.45.123.163 \
  "curl -s -o /dev/null -w 'HTTP %{http_code}\n' --max-time 3 http://127.0.0.1:8001/v1/models"
```

如果 `HTTP 000`，起一个：

```bash
ssh -p 2222 -i ~/.ssh/jupyterhub.pem mambauser@210.45.123.163 << 'EOF'
cd /home/mambauser/tangxuan/online_vllm
CUDA_VISIBLE_DEVICES=2 MODEL=/home/mambauser/tangxuan/models/Qwen3-VL-8B-Instruct-FP8 \
  PORT=8001 TP_SIZE=1 \
  nohup bash test/start_online_prefill_server.sh > test/logs/vllm_8001.log 2>&1 & disown
EOF
```

等 2–3 分钟，轮询直到 `HTTP 200`。模型 ID 确认：

```bash
ssh -p 2222 -i ~/.ssh/jupyterhub.pem mambauser@210.45.123.163 \
  "curl -s http://127.0.0.1:8001/v1/models | head -c 300"
```

> **警告 (坑 A)**：同一个端口不要起两个 vLLM！Linux 允许 `SO_REUSEPORT` 两个 uvicorn 都绑定成功，连接随机分发到某一个进程，online_prefill session 是进程内 dict，另一进程自然就 404。发现 `/v1/online_prefill/sessions/{id}` 间歇性 404 就先 `pgrep -f api_server` 看有没有两个进程在抢同一个端口。

### 3.2 云端：intake

**清掉旧实例后再启**（否则新实例绑 9100 失败）：

```bash
ssh -p 2222 -i ~/.ssh/jupyterhub.pem mambauser@210.45.123.163 << 'EOF'
for p in $(pgrep -f "python -m intake.server"); do kill -9 $p; done
sleep 2
cd /home/mambauser/tangxuan/online_vllm
env VLLM_API_BASE=http://127.0.0.1:8001 \
    BAVA_MAX_FRAMES_PER_WINDOW=4 \
    BAVA_STREAM_CONCURRENCY=2 \
    BAVA_MAX_QUEUED_WINDOWS=4 \
    nohup bash intake/start_intake.sh > intake/logs/intake.log 2>&1 & disown
EOF
```

> **坑 B**：`VLLM_API_BASE` 必须用 `env VAR=val` 前缀才能传进 `nohup bash` 的子 shell；直接 `VAR=val nohup ...` 在某些 ssh 非交互 shell 里会丢。
> **坑 C**：`start_intake.sh` 用 `${VAR:-default}` 形式读 env。如果 shell 里已有旧值，新值会被覆盖。每次 restart 想改参数就在 `env` 前缀里列全。

3–5 秒后确认活着：

```bash
ssh -p 2222 -i ~/.ssh/jupyterhub.pem mambauser@210.45.123.163 "curl -s http://127.0.0.1:9100/healthz"
# 期望: {"ok":true, ...}
```

日志一直在 `/home/mambauser/tangxuan/online_vllm/intake/logs/intake.log`（每次 restart 覆盖）。

### 3.3 本机：SSH 隧道

intake 的 9100 端口在公网被 Node Exporter 占，必须开隧道。**每次本机重启后都要重开**。也一并把 vLLM 的 :8001 转发下来方便调指标：

```bash
# 如果之前的隧道进程在，先清
pkill -f 'ssh -f -N -p 2222 -i /home/admin123/.ssh/jupyterhub.pem -L' 2>/dev/null

ssh -f -N -p 2222 -i ~/.ssh/jupyterhub.pem \
  -L 19100:127.0.0.1:9100 \
  -L 18001:127.0.0.1:8001 \
  mambauser@210.45.123.163

# 验证
curl -s http://127.0.0.1:19100/healthz | head -c 200
curl -s http://127.0.0.1:18001/v1/models | head -c 200
```

> **坑 D**：`curl http://210.45.123.163:9100` 不要用——那打到 Node Exporter 去了，返回的 HTML 页面里有 `HTTP 200` 但不是 intake。

---

## 4. 跑通验证

### 4.1 单流冒烟

用 UCF 的一个短片验证整条链：

```bash
export PYTHONPATH=/home/admin123/tangxuan
python3 -m edge.src.edge_main \
  --source edge/data/ucf/Abuse028_x264.mp4 \
  --cloud-ws-url ws://127.0.0.1:19100/stream \
  --stream-id smoke-01 \
  --rho 0.5 --window-seconds 2.0 --max-tokens 16 \
  --pace-realtime --linger-s 30
```

预期看到：

```
[edge] seen=NN sel=MM ... rho=0.500 win=1 ... rate=25.0 pps      # 25fps 实时节奏
[edge-uplink] rho update 0.500 -> 0.xxx reason='...'              # 控制器在下发
[edge-uplink] result window=0 text='...'                          # vLLM 返回文本
[edge] done ...
```

日志没 `result` 行 = 云端没闭环。常见原因：vLLM 不通（查 :8001）、intake 的 controller 挂了（查 `/healthz`）、tunnel 断了（curl 测）。

### 4.2 反向通道测试

另起终端推个 ρ 下来：

```bash
curl -s -X POST http://127.0.0.1:19100/admin/push_rho \
  -H 'Content-Type: application/json' \
  -d '{"stream_id":"smoke-01","rho":0.1,"alpha":0.25,"reason":"manual"}'
```

运行中的 edge 终端应立即打印 `rho update 0.xxx -> 0.100 reason='manual'`。如果没有，说明 uplink 后台线程挂了或 controller 没开。

### 4.3 多流基准（主要压测入口）

```bash
export PYTHONPATH=/home/admin123/tangxuan
python3 -m edge.tools.bench \
  --n 4 --duration 60 \
  --cloud-ws-url ws://127.0.0.1:19100/stream \
  --intake-admin-base http://127.0.0.1:19100 \
  --sources 'edge/data/ucf/*.mp4' \
  --rho 0.5 --window-seconds 2.0 --max-tokens 12 \
  --probe-interval 0.5 \
  --out /tmp/bench_n4
```

产出：

```
/tmp/bench_n4/
  manifest.json         # 每路 edge 的 cmdline + pid + 源映射
  probes.jsonl          # 每 500ms 一行（Q/KV/运行数/每流 ρ/α）
  summary.json          # append/e2e 的 P50/P95
  edge-bench-00.log     # 每路 edge 的 stdout+stderr
```

控制台最后会打印聚合 P95，例如：

```
aggregate latency: n=348 append p50=1157ms p95=3382ms e2e p50=101774ms p95=230213ms
```

> **跑完后**：bench harness 会 SIGTERM+SIGKILL 所有 edge 子进程。万一有漏的，手动 `pkill -9 -f edge.src.edge_main`。

---

## 5. 观测点（按排错顺序）

| 查什么 | 怎么查 |
|---|---|
| 云端 vLLM 活吗？有几帧被 prefill 过？ | `curl -s http://127.0.0.1:18001/metrics \| grep -E '^vllm:(num_requests|kv_cache|prompt_tokens)_total'` |
| intake 活吗？跟踪的流有哪些？ρ/α 当前值？ | `curl -s http://127.0.0.1:19100/healthz \| python3 -m json.tool` |
| 最近窗口的延迟分布 | `curl -s http://127.0.0.1:19100/stats/latency \| python3 -m json.tool` |
| 控制器每 tick 的时间序列（画 Fig.D）| 云端 `/tmp/bava_controller.jsonl`（每 tick 一行 JSON）|
| 每包 anchor embedding 历史 | 云端 `/tmp/bava_anchors.jsonl` |
| intake 最近 log | 云端 `tail -f /home/mambauser/tangxuan/online_vllm/intake/logs/intake.log` |
| vLLM 最近 log | 云端 `tail -f /home/mambauser/tangxuan/online_vllm/test/logs/vllm_8001.log` |
| edge 单流内部状态 | 运行时 stderr 每 100 包一行 `[edge] seen= sel= rho= win= sent= dropped=`|

---

## 6. 环境变量总表

### 6.1 intake（云端 `start_intake.sh`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_API_BASE` | `http://127.0.0.1:8000` | vLLM HTTP 端点。**现用 :8001** |
| `INTAKE_HOST` / `INTAKE_PORT` | `0.0.0.0` / `9100` | intake 自身监听 |
| `BAVA_LOG` | `info` | 日志级别 |
| `BAVA_CONTROLLER_ENABLED` | `1` | 0 = 关 M5 控制器，进入静态 ρ/α 模式 |
| `BAVA_TICK_S` | `0.5` | 控制器 tick 周期 |
| `BAVA_Q_TARGET` | `2.0` | prefill 队列长设定点 Q* |
| `BAVA_KV_TARGET` | `0.6` | KV 占用率设定点 KV* |
| `BAVA_MU_RHO` / `BAVA_MU_ALPHA` | `0.05` | M5 步长 |
| `BAVA_Q_DEADBAND` / `BAVA_KV_DEADBAND` | `1.0` / `0.05` | 误差死区 |
| `BAVA_MIN_UPDATE_S` | `1.0` | 每流最小 ρ 下发间隔（防刷 WS）|
| `BAVA_PREEMPT_PANIC` | `0.5` | 抢占率阈值（每秒），超了额外扣 ρ |
| `BAVA_PREEMPT_STEP` | `0.1` | 紧急刹车额外步长 |
| `BAVA_MAX_FRAMES_PER_WINDOW` | `16` | 窗口内帧数封顶（超出 uniform 抽样）|
| `BAVA_STREAM_CONCURRENCY` | `2` | **每流**同时在 vLLM 上的窗口数 |
| `BAVA_MAX_QUEUED_WINDOWS` | `8` | **每流**待处理窗口上限；超出丢最老 |
| `BAVA_ALPHA_WEIGHTED` | `0` | 1 = 按 anchor.score 给每帧分不同 α |
| `BAVA_ALPHA_MIN_SIDE` / `MAX_SIDE` | `112` / `1568` | α 缩放的像素下界/上界 |
| `BAVA_ALPHA_ALIGN` | `28` | ViT patch 对齐（Qwen3-VL = 14×2）|
| `BAVA_ALPHA_SCORE_POWER` | `1.0` | weighted 模式下 score 锐化指数 |
| `BAVA_ALPHA_PER_FRAME_FLOOR` | `0.15` | weighted 模式每帧 α 下限 |
| `BAVA_CONTROLLER_LOG` | `/tmp/bava_controller.jsonl` | 时间序列落盘路径 |
| `BAVA_ANCHOR_LOG` | `/tmp/bava_anchors.jsonl` | anchor 透传落盘路径 |
| `BAVA_METRICS_BASE` | = `VLLM_API_BASE` | 控制器 scraper 的 metrics 端点，可单独指向另一 vLLM |

### 6.2 边缘（CLI 参数）

```
python3 -m edge.src.edge_main --help
```

关键：
- `--source` RTSP URL 或本地 mp4 路径
- `--cloud-ws-url` `ws://127.0.0.1:19100/stream`（走隧道）
- `--stream-id` 每路相机唯一 ID（bench 自动给 `bench-NN`）
- `--rho` 起始保帧率 [0.02, 1.0]
- `--window-seconds` 窗口时长（默认 4.0）
- `--max-tokens` 对应 vLLM `max_tokens`，默认 24
- `--prompt` 送给 vLLM 的 user prompt
- `--model` 必须等于 vLLM 当前装载的模型路径
- `--pace-realtime` 按 pts_s 节流（模拟真实相机），benchmark 必开
- `--loop-source` EOF 后重开，pts 累加偏移，长基准必开
- `--linger-s` source EOF 后等云端返回 result 的时间（默认 30s）
- `--rtsp-transport` `tcp` / `udp`（RTSP 才用到）

---

## 7. 文件快速索引

### 7.1 边缘 `/home/admin123/tangxuan/edge/`

```
src/
  config.py            # EdgeConfig dataclass
  rho_state.py         # 线程安全可变 ρ
  wire.py              # WS 协议 (text/binary)
  uplink.py            # WS 客户端（后台 asyncio 线程）
  rtsp_source.py       # PyAV demux，AVCC→AnnexB，SPS/PPS 注入
  h264_parser.py       # NAL 游走 + slice_type
  features.py          # 包级特征 + 16 维 placeholder anchor
  scorer.py            # PacketGame 风格评分
  gop_buffer.py        # GOP 对齐的 decoder-safe 选择
  edge_main.py         # CLI 入口 + _paced_iter (pts 节流 + loop)
tools/
  bench.py             # N 流 benchmark harness
  cloud_sink.py        # 本地冒烟用的 TCP 接收器（不是当前云端路径）
  make_test_clip.py    # 生成 edge/data/test.mp4 的工具
data/
  test.mp4             # 5s 合成测试片
  ucf/*.mp4            # 10 个 UCF 样本（448×448 @25fps GOP=25）
  bench_runs/          # 基准运行归档
```

### 7.2 云端 `/home/mambauser/tangxuan/online_vllm/intake/`

```
server.py              # FastAPI + websockets 主入口
wire.py                # edge wire.py 的对称副本
window_assembler.py    # 窗口聚包 → 解码 → α → JPEG → vLLM
gop_decoder.py         # decode_to_bgr + encode_bgr_to_jpeg_data_uri
alpha_executor.py      # α 执行器（uniform / anchor-weighted）
vllm_client.py         # httpx.AsyncClient 封 vLLM online_prefill
metrics.py             # Prometheus scraper
controller.py          # BAVA M5 控制器
latency.py             # per-stream P50/P95 追踪
bava_stub.py           # 已弃用，保留不删
start_intake.sh        # 启动脚本
logs/intake.log        # 运行时日志
```

---

## 8. 已知不足 / 待办

按紧急程度排序。每条都写清现象、根因、解法方向，下次谁接手直接上手。

**2026-04-24 更新**：8.2 / 8.3 / 8.4 / 8.6 已修复（阶段 5）。N=4 e2e_p95 从 230s 降到 22s（10×），N=8 能跑通不再冻结。详见 [debug_report.md §8](debug_report.md#8-阶段-5--问题修复2026-04-24)。剩余问题下面仍保留。

### 8.1 ~~BAVA vs 静态基线的 A/B 跑不干净~~ ✅ 已修复（阶段 6）

intake 现在自己 track 创建过的 rids（`IntakeState.active_rids`），在 FastAPI shutdown / SIGTERM 时 `DELETE /v1/online_prefill/sessions/{id}` 一把清掉。加了 `POST /admin/purge_sessions` 作为手动触发。`edge/tools/ab_bench.py` 用这套机制做 A/B，不用重启 vLLM，每切 config 只需 ~10s 重启 intake。

**实测 N=4 60s**（vLLM :8003 FP8 单实例，UCF 轮询）：

| config | windows | append_p95 | e2e_p95 |
|---|---:|---:|---:|
| static_full (ρ=1, α=1) | 72 | 805ms | 21674ms |
| static_half (ρ=0.5, α=0.5) | 89 | 823ms | 17061ms |
| bava_dynamic | 90 | 757ms | **14546ms** |

BAVA vs static_full e2e_p95 -33%，vs static_half -15%。详见 debug_report §9。

### 8.2 ~~intake 的 PyAV 解码阻塞 event loop~~ ✅ 已修复（阶段 5）

PyAV decode / α resize / JPEG encode 全部走 `asyncio.to_thread`。N=4 e2e_p95 230s → 22s（10.6×），N=8 不再冻结。代码位置：`intake/window_assembler.py`。

### 8.3 ~~edge SIGINT 不干净~~ ✅ 已修复（阶段 5）

`edge_main.py` 入口注册 SIGINT/SIGTERM handler → `STOP` threading.Event，`_paced_iter` 每轮检查，`time.sleep` 换成 `STOP.wait(timeout=)`。`uplink.close()` 在停止路径用短超时（drain=1.5s, linger=0.2s）。SIGTERM → exit ~5s（旧 9s，且曾需 SIGKILL 兜底）。

### 8.4 ~~控制器是全流广播，没有 per-stream 加权~~ ✅ 已修复（阶段 5）

`BavaController.__init__` 新增 `load_lookup: Callable[[stream_id], (inflight_appends, last_append_ms, inflight_windows)]`。每 tick 算权重 `w_i = clamp(load_i / mean_load, [0.5, 2.0])`；cutoff 用 w，climb-back 保持 w=1.0 均匀恢复。env 钮：`BAVA_PER_STREAM_WEIGHTING` / `BAVA_STREAM_WEIGHT_MIN` / `BAVA_STREAM_WEIGHT_MAX`。

仍未做的部分：
- SLA 优先级（低优流先砍）
- vLLM metrics 带 stream label（需要改 vllm/ — 违反约束）

### 8.5 α 执行器是近似，不是严格 token-level prune

**现象**：理论上 α=0.25 应该让 visual token 数降到 ¼，实际因为 Qwen3-VL processor 里还有 `min_pixels/max_pixels` clip、dynamic resolution 归一化，√α resize 只是近似。
**根因**：我们没改 vLLM，只能在 pixel 空间执行 α。
**修法**：在 vLLM 的 visual encoder 输出端 hook，直接丢 (1-α) 份 embedding token。需要修改 `vllm/entrypoints/serve/online_prefill/api_router.py` 里的 `_prepare_append_streaming_input`。

### 8.6 ~~M5 控制器没有不对称恢复~~ ✅ 已修复（阶段 5）

`controller.py` 新增 climb-back：Q/KV 连续 `BAVA_CLIMB_BACK_TICKS=6` 个 tick 低于 `target - slack` 后，以小步长 (`BAVA_CLIMB_BACK_STEP_RHO=0.02`) 爬回 ρ/α。验证：ρ=0.9 起跑，Q=0 保持 → climb-back 触发 → 0.9→0.92→0.94→0.96→0.98→1.0。

更严格 MPC 形式还没上（把 Acc 约束塞进 QP），保留为后续。

### 8.7 anchor embedding 仍是 placeholder

**现象**：`edge/src/features.py:anchor_embedding` 是 16 维手工特征（IDR/slice_type/size/iat 之类）。`apply_alpha_weighted` 接好了但拿这个 score 加权差异微弱。
**根因**：没训模型。
**修法**：
1. 用 UCF 帧（或其他数据集）训一个轻量教师→学生（CLIP embedding 蒸馏到小 MLP）
2. 导出 TensorRT FP16 跑在 Jetson iGPU 上
3. 边缘侧替换 `anchor_embedding` 的返回值，header 里带过去就行（已有管道）

### 8.8 scale-up 部分完成（N=4, N=8）；N≥16 待做

**已验证**：
- N=4 60s：append_p95=700ms, e2e_p95=22s, 103 windows。
- N=8 60s：append_p95=2484ms, e2e_p95=35s, 237 windows，不冻结。

**仍未做**：
- N=16/32/64/128 — 预期 N≥16 在当前 Qwen3-VL-8B-Instruct-FP8 / tp=1 / 2K ctx 会把 KV 打满，需要换大 ctx 或 tp=2 实例。
- A/B 基线矩阵（ρ=1/α=1, ρ=0.5/α=0.5 静态双轴 vs BAVA 动态）— 需要解决 (8.1) 先。
- 300s duration 长跑 — 现在测试都是 60s，瞬态可能不充分。

### 8.9 Fig.E 消融还没做

技术路线 §6.5 要 Fig.E 消融四项：关 controller / 关 online prefill / 关语义锚点 / 关 Prop.2 驱动的适配。目前：
- `BAVA_CONTROLLER_ENABLED=0` ✅
- 关 online prefill：要改 intake 不走 online_prefill 而走 chat.completions + 整批帧（`compare_online_prefill_latency.py` 的 native_chat 分支可参考）
- 关语义锚点：`BAVA_ALPHA_WEIGHTED=0` + 在 α 分配里不消费 anchor → 已是默认
- 关 Prop.2 适配：`BAVA_MU_RHO=0 BAVA_MU_ALPHA=0` 让控制器空转 → 已支持

### 8.10 多 GPU / 多引擎支持缺失

**现象**：`BavaController` 只拉一个 `/metrics` 端点（全局 vLLM 状态）。如果将来要多台 vLLM 负载分担，controller 不知道怎么聚合。
**修法**：controller 接受 N 个 scraper + 引擎级路由函数（按 stream_id → engine 哈希）。没着急做。

### 8.11 intake 的进程管理依赖 nohup，生产上不合适

生产应该给 intake 写 systemd unit（`intake.service`），或跑在容器里。现在 `kill -9` + `nohup bash` 的套路只是开发用。

---

## 9. 备忘：一些现场经验

- SSH 非交互 shell 里 `pkill -f 'intake.server'` 会把 SSH 自己这条命令也命中（因为 `grep 'intake.server'` 匹配 bash -c 的 cmdline）。用 `for p in $(pgrep -f 'python -m intake.server'); do kill $p; done` 避开。
- 本机 ssh key 必须 `chmod 600 ~/.ssh/jupyterhub.pem`，否则 OpenSSH 拒绝加载。
- uv 创建的 venv 里没有 pip；装包用 `uv pip install`，不要用 `python -m pip`。
- FastAPI `@app.on_event("startup")` 是 deprecated 的，后续可以迁到 `lifespan`，现在能跑就不动。
- 云端 210.45.123.163 是多用户共享机，nvidia-smi 上看到的 `[Not Found]` 进程多半是别人的，不要动。
- 边缘 `edge/data/test.mp4` 只有 5 秒，不开 `--pace-realtime` 跑完全流要 0.1 秒，α 下发来不及生效——这是 **控制器看起来没在 work 的最常见假象**。出现疑问先确认 pacing 开了。
