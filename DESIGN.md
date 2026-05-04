# Linesight Improvement Plan

## Goal

Scale Linesight from its current 6.5M-parameter IQN agent to a significantly larger model,
improve sample efficiency with algorithmic upgrades, and enable distributed training
(game collection on Windows, training on DGX0 with A100 GPUs).

## Current Baseline

| Property | Value |
|----------|-------|
| Architecture | IQN (Implicit Quantile Networks) + Dueling |
| Parameters | 6,578,333 |
| Visual backbone | 4-layer CNN (1→16→32→64→32), output 5632-dim |
| Float features | 184-dim → 2-layer MLP → 256-dim |
| Dense heads | 5888 → 512 (A-head + V-head, dueling) |
| IQN quantiles | 8 training, 32 inference |
| Actions | 12 discrete (steering × accel/brake combos) |
| Observations | 120×160 grayscale + 184 floats per step |
| Action interval | 50ms |
| Replay buffer | 50K–200K transitions (staircase schedule) |
| Batch size | 512 |
| Training VRAM | ~2.7 GB estimated |
| Inference VRAM | ~53 MB |

## Hardware

| Machine | Role | Specs |
|---------|------|-------|
| Windows laptop | Game collection + current training | 32GB RAM, NVIDIA A2000 (4-6GB VRAM) |
| DGX0 (Chapman) | Future training server | 8× A100-80GB, 2TB RAM, Ubuntu 22.04, Docker only |

## Phases

### Phase 0: Baseline + Low-Hanging Fruit (laptop only)

**Goal:** Establish a reproducible baseline, then apply config-only improvements.

#### 0A — Establish baseline
- Train on ESL-Hockolicious with current config for ~3M frames
- Record: race completion times, loss curve, Q-value stats, grad norms
- **Expected time:** ~3–5 hours on laptop (50ms/action, 2 collectors)
- **Success:** Agent completes the map. Record best race time.

#### 0B — Enable PER (alpha=0.2)
- **Change:** `config.py` line 97: `prio_alpha = np.float32(0.2)`
- **Risk:** Very low. Full PER implementation exists, tested in original codebase.
- **Expected effect:** 10–30% improvement in sample efficiency (faster convergence to same race time)
- **Test:** Train 3M frames, compare loss curve and race times to baseline
- **Rollback:** Set alpha back to 0

#### 0C — Enable data augmentation
- **Change:** `config.py` line 161: `apply_randomcrop_augmentation = True`
- **Risk:** Very low. Random crop ±2 pixels with edge padding, same crop for state+next_state.
- **Expected effect:** Modest regularization, more robust visual features. May help or be neutral.
- **Test:** Train 3M frames with PER still on, compare to 0B
- **Rollback:** Set flag back to False

#### 0D — Batch size warmup (optional)
- **Change:** Add schedule to start at batch_size=64, ramp to 512 over first 500K frames
- **Risk:** Low. Requires minor code change in learner_process.py.
- **Expected effect:** Faster early learning (smaller batches = more updates per frame)
- **Test:** Compare early training curve (first 500K frames) to 0B/0C

**Checkpoint after Phase 0:** We should have a measurably better agent than the current default config,
with race times and training curves to prove it. Total time: ~1–2 days of training.

---

### Phase 1: Exploration Improvements (laptop only)

**Goal:** Replace epsilon-greedy with learned exploration.

#### 1A — NoisyNet
- **Change:** Replace `nn.Linear` layers in A_head and V_head with `NoisyLinear`
  (factorized Gaussian noise, ~100 lines of new code in `iqn.py`)
- Disable epsilon-greedy and Boltzmann exploration (`epsilon_schedule` → all zeros)
- **Risk:** Medium. Changes exploration dynamics. Must verify gradient flow through noise params.
- **Expected effect:** More directed exploration — the agent learns *where* it's uncertain.
  Literature shows consistent improvement over epsilon-greedy in value-based methods.
- **Test:** Train 3M frames. Compare:
  - Race times (should match or beat Phase 0 result)
  - Action entropy over time (should decrease as agent becomes more certain)
  - Noise parameter magnitudes (should shrink in well-learned states)
- **Rollback:** Revert to epsilon-greedy (keep NoisyLinear code behind a config flag)

**Checkpoint after Phase 1:** Better exploration without manual epsilon schedules.
Total time: ~1 day implementation + 1 day training.

---

### Phase 2: Distributed Training Infrastructure

**Goal:** Decouple game collection from training so we can scale the model.

#### 2A — Network communication layer
- **Change:** Replace `multiprocessing.Queue` between collector and learner with ZMQ sockets
- Collector (Windows laptop): sends rollouts, receives weight updates
- Learner (DGX0 Docker container): receives rollouts, sends weights
- **Architecture:**
  ```
  Windows laptop (VPN)              DGX0 (Docker, 1× A100)
  ┌──────────────────┐              ���──────────────────────┐
  │ Trackmania × 2   │  rollouts →  │  Learner process     │
  │ Collector process │ ── ZMQ ───→ │  Training loop       │
  │ (inference only)  │  ← weights  │  Replay buffer       │
  └────────────���─────┘              └────────��─────────────┘
  ```
- Rollout size: ~50–100 KB per mini-race. Sent every ~7 seconds. Trivial bandwidth.
- Weight sync: ~26 MB (current model) every N batches. Fine even over VPN.
- **Key files to modify:**
  - `scripts/train.py` — split into `train_learner.py` and `train_collector.py`
  - `trackmania_rl/multiprocess/learner_process.py` — accept rollouts from ZMQ instead of mp.Queue
  - `trackmania_rl/multiprocess/collector_process.py` — send rollouts to ZMQ, receive weights
  - New: `trackmania_rl/network_transport.py` — serialization + ZMQ wrapper
- **Risk:** Medium-high. Distributed systems have failure modes (disconnects, stale weights, etc.).
  Need heartbeat/reconnect logic.
- **Test:**
  1. First test locally: run learner and collector as separate processes on the same machine,
     communicating over localhost ZMQ. Verify identical training curves to Phase 0/1.
  2. Then test over network: collector on laptop, learner on DGX0.
  3. Verify: no training regression, acceptable latency (<1s for weight updates)
- **Expected time:** 2–3 days implementation, 1 day testing

#### 2B — Docker container for DGX0
- Build a Docker image with: Python 3.10, PyTorch + CUDA 12.4, torchrl 0.2.1, ZMQ, project code
- Mount `/nfshome/jdowd` for checkpoints and tensorboard logs
- Claim 1 GPU (check `nvidia-smi` first per DGX0 rules)
- **Dockerfile + launch script**

**Checkpoint after Phase 2:** Training runs on A100, collection on laptop. Same model, same
performance, but now we can scale up. Total time: ~3–4 days.

---

### Phase 3: Scale the Model

**Goal:** Use the A100's 80GB VRAM to train a much larger network.

#### 3A — Larger visual backbone
- **Change:** Scale conv channels from (1,16,32,64,32) to (1,64,128,256,128) or add residual blocks
- Add batch normalization or layer normalization to conv layers
- `conv_head_output_dim` increases from 5632 to ~30K+
- **Risk:** Medium. Larger model needs more data to converge. May need to increase buffer size
  and reduce learning rate.
- **Test:** Train 5–10M frames (larger model needs more data). Compare race times.
- **Expected time:** ~12–24 hours training on A100

#### 3B — Frame stacking (2–4 frames)
- **Change:** Input channels from 1 → N. Buffer stores N consecutive frames.
- Gives the agent velocity/acceleration information from raw pixels
- **Risk:** Medium-high. Many tensor shape changes across codebase. Buffer memory increases N×.
- **Key touch points:** `iqn.py` (input channels), `buffer_management.py` (transition assembly),
  `buffer_utilities.py` (collate), `collector_process.py` (frame history), `config.py` (new param)
- **Test:** Train 5M frames. Watch for: does the agent learn to use temporal info?
  Compare Q-value variance (should decrease — less partial observability).

#### 3C — Wider dense layers
- Scale `dense_hidden_dimension` from 1024 → 2048 or 4096
- Scale `float_hidden_dim` from 256 → 512
- **Risk:** Low given A100 VRAM. Just config changes + verified shapes.
- **Test:** Train 5M frames, compare to 3A.

#### 3D — More IQN quantiles
- Increase `iqn_n` from 8 → 16 or 32 (training), `iqn_k` from 32 → 64 (inference)
- More quantiles = better distributional approximation = better risk-sensitive decisions
- **Risk:** Low. Linear increase in forward pass compute. Already well within A100 budget.
- **Test:** Train 5M frames. Q-value distribution should be smoother.

**Checkpoint after Phase 3:** Significantly larger model. Expect noticeably better race times
if the current model is capacity-limited. Total time: ~1 week of experimentation.

---

### Phase 4: Advanced Techniques (if Phase 3 shows gains)

#### 4A — Auxiliary representation losses
- Add reward prediction or next-state prediction heads to the shared backbone
- Forces richer learned representations
- **Risk:** Medium. Must balance auxiliary loss weight to avoid dominating IQN loss.

#### 4B — Spectral normalization
- Apply `torch.nn.utils.spectral_norm` to conv and dense layers
- Stabilizes training with larger models
- **Risk:** Low to try, unclear benefit.

#### 4C — Curriculum learning
- Start on easy maps (A01-Race), gradually introduce harder maps
- Infrastructure already exists in `map_cycle`
- **Risk:** Low. Config-only change.

#### 4D — Multi-GPU training (if needed)
- Use DDP across multiple A100s if single-GPU is too slow
- Only worth it for very large models (>100M params)

---

## Testing Protocol

**Between every phase change:**

1. Train for the specified number of frames
2. Record to tensorboard:
   - Race completion time (primary metric)
   - Training loss curve
   - Gradient norms (per-layer)
   - Q-value mean and std
   - Epsilon / exploration rate
   - Buffer size and utilization
   - PER priority distribution (if enabled)
3. Compare against the previous phase's tensorboard logs
4. **Keep checkpoints** — save model weights at end of each phase for A/B comparison

**Regression criteria:** If race times are >10% worse after 3M frames, roll back the change.

## Estimated Timeline

| Phase | Work | Training | Total |
|-------|------|----------|-------|
| 0: Baseline + easy wins | 1 hour | 1–2 days | 2 days |
| 1: NoisyNet | 1 day | 1 day | 2 days |
| 2: Distributed infra | 3 days | 1 day | 4 days |
| 3: Scale model | 2 days | 1 week | 9 days |
| 4: Advanced | 3 days | 1 week | 10 days |
| **Total** | | | **~4 weeks** |

Note: Phases 0–1 can run on the laptop. Phase 2+ requires the DGX0 connection.
Phases are sequential — each builds on the previous.

## File Map (key files to modify)

| File | Phase | Change |
|------|-------|--------|
| `config_files/config.py` | 0, 1, 3 | Hyperparameters, flags, schedules |
| `trackmania_rl/agents/iqn.py` | 1, 3 | NoisyLinear, larger backbone, more quantiles |
| `trackmania_rl/multiprocess/learner_process.py` | 0D, 2 | Batch warmup, ZMQ integration |
| `trackmania_rl/multiprocess/collector_process.py` | 2 | ZMQ integration |
| `trackmania_rl/buffer_management.py` | 3B | Frame stacking transition assembly |
| `trackmania_rl/buffer_utilities.py` | 3B | Frame stacking collate |
| `scripts/train.py` | 2 | Split into learner/collector entry points |
| NEW: `trackmania_rl/network_transport.py` | 2 | ZMQ serialization layer |
| NEW: `Dockerfile` | 2B | DGX0 container |
