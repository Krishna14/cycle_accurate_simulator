#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
 
namespace sim {
 
static inline uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
  return (a + b - 1) / b;
}
 
struct SimConfig {
  // System
  uint32_t num_cus = 4;
  uint32_t num_dram_channels = 4;
  uint32_t max_cycles = 100000000;
 
  // NoC (simple crossbar-ish model with per-channel output links)
  uint32_t noc_hop_latency_cycles = 2;
  uint32_t noc_bytes_per_cycle = 32; // per output link
  uint32_t noc_injection_fifo_depth = 64;
  uint32_t noc_ejection_fifo_depth = 64;
 
  // DRAM
  uint32_t dram_latency_cycles = 200;
  uint32_t dram_bytes_per_cycle = 64; // per channel data bus
  uint32_t dram_req_fifo_depth = 256;
  uint32_t dram_rsp_fifo_depth = 256;
 
  // CU compute models
  uint64_t cu_flops_per_cycle = 4096;      // GEMM compute throughput
  uint32_t cu_vec_elems_per_cycle = 64;    // softmax vector throughput
  uint32_t cu_sfu_elems_per_cycle = 32;    // exp throughput
 
  // Data type sizes (bytes)
  uint32_t gemm_elem_bytes = 2;     // fp16
  uint32_t gemm_accum_bytes = 4;    // fp32 accumulate -> fp16 output modeled as write bytes
  uint32_t softmax_elem_bytes = 2;  // fp16
};
 
// A bounded FIFO with *registered* behavior:
// - Consumers pop from the "current" queue.
// - Producers push into an "incoming" queue that becomes visible next cycle via commit().
template <typename T>
class RegFifo {
 public:
  explicit RegFifo(size_t capacity = 0) : cap_(capacity) {}
 
  void set_capacity(size_t c) { cap_ = c; }
  size_t capacity() const { return cap_; }
  size_t size_curr() const { return curr_.size(); }
  size_t size_incoming() const { return incoming_.size(); }
  bool empty_curr() const { return curr_.empty(); }
  bool empty_all() const { return curr_.empty() && incoming_.empty(); }
 
  bool can_push_next() const { return (curr_.size() + incoming_.size()) < cap_; }
  bool push_next(const T& v) {
    if (!can_push_next()) return false;
    incoming_.push_back(v);
    return true;
  }
  bool push_next(T&& v) {
    if (!can_push_next()) return false;
    incoming_.push_back(std::move(v));
    return true;
  }
 
  bool can_pop_curr() const { return !curr_.empty(); }
  std::optional<T> pop_curr() {
    if (curr_.empty()) return std::nullopt;
    T v = std::move(curr_.front());
    curr_.pop_front();
    return v;
  }
 
  const T& front_curr() const { return curr_.front(); }
 
  void commit() {
    // Items pushed this cycle become visible next cycle.
    while (!incoming_.empty()) {
      curr_.push_back(std::move(incoming_.front()));
      incoming_.pop_front();
    }
  }
 
 private:
  size_t cap_ = 0;
  std::deque<T> curr_;
  std::deque<T> incoming_;
};
 
enum class PacketType : uint8_t {
  ReadReq = 0,
  WriteReq = 1,
  ReadResp = 2,
  WriteAck = 3,
};
 
struct Packet {
  PacketType type{};
  uint32_t src_cu = 0;
  uint32_t dst_dram_chan = 0;
  uint64_t addr = 0;
  uint32_t bytes = 0;
  uint64_t task_id = 0;
  uint32_t tag = 0; // opaque subtag (operand kind, phase, etc.)
};
 
enum class OpType : uint8_t { Gemm = 0, Softmax = 1 };
 
struct GemmParams {
  uint32_t mt = 0, nt = 0, kt = 0;
  uint32_t m0 = 0, n0 = 0; // tile origin
  uint32_t k_total = 0;
};
 
struct SoftmaxParams {
  uint32_t row = 0;
  uint32_t lt = 0;     // tile length
  uint32_t l_total = 0;
};
 
struct TileTask {
  uint64_t id = 0;
  OpType op{};
  uint64_t enqueue_order = 0; // for debugging
  // Dependencies (operator-level): task is runnable once all deps completed.
  std::vector<uint64_t> deps;
 
  // Operand base addresses (synthetic). For operator-level modeling these can just be unique IDs.
  uint64_t a_base = 0, b_base = 0, c_base = 0;
 
  GemmParams gemm{};
  SoftmaxParams softmax{};
};
 
struct Workload {
  std::vector<TileTask> tasks;
};
 
class TaskGraph {
 public:
  explicit TaskGraph(const std::vector<TileTask>& tasks) {
    for (auto& t : tasks) {
      tasks_[t.id] = t;
      remaining_deps_[t.id] = static_cast<uint32_t>(t.deps.size());
      for (auto dep : t.deps) {
        dependents_[dep].push_back(t.id);
      }
    }
    total_ = tasks.size();
  }
 
  uint64_t total_tasks() const { return total_; }
  uint64_t completed_tasks() const { return completed_; }
 
  bool is_done() const { return completed_ == total_; }
 
  // Call when a task completes; returns newly-unblocked task IDs.
  std::vector<uint64_t> mark_done(uint64_t task_id) {
    std::vector<uint64_t> newly_ready;
    auto it = dependents_.find(task_id);
    if (it != dependents_.end()) {
      for (auto dep_id : it->second) {
        auto& cnt = remaining_deps_[dep_id];
        if (cnt > 0) cnt--;
        if (cnt == 0) newly_ready.push_back(dep_id);
      }
    }
    completed_++;
    return newly_ready;
  }
 
  bool is_ready(uint64_t task_id) const {
    auto it = remaining_deps_.find(task_id);
    if (it == remaining_deps_.end()) return false;
    return it->second == 0;
  }
 
  const TileTask& get(uint64_t task_id) const { return tasks_.at(task_id); }
 
 private:
  uint64_t total_ = 0;
  uint64_t completed_ = 0;
  std::unordered_map<uint64_t, TileTask> tasks_;
  std::unordered_map<uint64_t, uint32_t> remaining_deps_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> dependents_;
};
 
// Simple round-robin scheduler that dispatches ready tasks into per-CU input FIFOs (registered).
class Scheduler {
 public:
  Scheduler(uint32_t num_cus, TaskGraph* graph, std::vector<RegFifo<TileTask>>* cu_task_in)
      : num_cus_(num_cus), graph_(graph), cu_task_in_(cu_task_in) {}
 
  void seed_initial_ready(const std::vector<TileTask>& tasks) {
    for (auto& t : tasks) {
      if (graph_->is_ready(t.id)) ready_.push_back(t.id);
    }
  }
 
  void on_newly_ready(const std::vector<uint64_t>& ids) {
    for (auto id : ids) ready_.push_back(id);
  }
 
  void tick_dispatch() {
    // Try to dispatch as many ready tasks as queues allow, in round-robin CU order.
    // This models a centralized scheduler that can enqueue 1+ tasks/cycle; if you want a hard
    // per-cycle issue width, cap it here.
    size_t i = 0;
    while (i < ready_.size()) {
      uint32_t cu = rr_cu_++ % num_cus_;
      auto& q = (*cu_task_in_)[cu];
      if (!q.can_push_next()) {
        // Try other CUs before giving up this cycle.
        bool placed = false;
        for (uint32_t k = 0; k < num_cus_; k++) {
          uint32_t cu2 = (cu + 1 + k) % num_cus_;
          if ((*cu_task_in_)[cu2].can_push_next()) {
            TileTask t = graph_->get(ready_[i]);
            (*cu_task_in_)[cu2].push_next(std::move(t));
            placed = true;
            break;
          }
        }
        if (!placed) break; // no space anywhere
      } else {
        TileTask t = graph_->get(ready_[i]);
        q.push_next(std::move(t));
      }
      ready_.erase(ready_.begin() + static_cast<long>(i));
    }
  }
 
 private:
  uint32_t num_cus_ = 0;
  uint32_t rr_cu_ = 0;
  TaskGraph* graph_ = nullptr;
  std::vector<RegFifo<TileTask>>* cu_task_in_ = nullptr;
  std::vector<uint64_t> ready_;
};
 
class NoC {
 public:
  struct LinkInFlight {
    bool valid = false;
    Packet pkt{};
    uint32_t remaining_latency = 0;
    uint32_t remaining_bytes = 0;
  };
 
  NoC(uint32_t num_cus,
      uint32_t num_dram_channels,
      uint32_t hop_latency_cycles,
      uint32_t bytes_per_cycle)
      : num_cus_(num_cus),
        num_ch_(num_dram_channels),
        hop_lat_(hop_latency_cycles),
        bpc_(bytes_per_cycle) {
    // One output link per DRAM channel (CU->DRAM direction)
    cu_to_mc_link_.resize(num_ch_);
    rr_in_sel_cu_to_mc_.assign(num_ch_, 0);
    // One output link per CU (DRAM->CU direction)
    mc_to_cu_link_.resize(num_cus_);
    rr_in_sel_mc_to_cu_.assign(num_cus_, 0);
  }
 
  // These FIFOs live outside but are connected here:
  // - CU inject: cu_out[i] -> NoC
  // - MC eject: noc_to_mc[ch] -> MC
  // - MC inject: mc_out[ch] -> NoC
  // - CU eject: noc_to_cu[i] -> CU
  void tick(std::vector<RegFifo<Packet>>* cu_out,
            std::vector<RegFifo<Packet>>* noc_to_mc,
            std::vector<RegFifo<Packet>>* mc_out,
            std::vector<RegFifo<Packet>>* noc_to_cu) {
    tick_dir_cu_to_mc(*cu_out, *noc_to_mc);
    tick_dir_mc_to_cu(*mc_out, *noc_to_cu);
  }
 
  bool idle() const {
    for (auto& l : cu_to_mc_link_) {
      if (l.valid) return false;
    }
    for (auto& l : mc_to_cu_link_) {
      if (l.valid) return false;
    }
    return true;
  }
 
  uint64_t cu_to_mc_busy_cycles() const { return cu_to_mc_busy_cycles_; }
  uint64_t mc_to_cu_busy_cycles() const { return mc_to_cu_busy_cycles_; }
 
 private:
  void tick_dir_cu_to_mc(std::vector<RegFifo<Packet>>& cu_out, std::vector<RegFifo<Packet>>& noc_to_mc) {
    // For each MC channel output, advance serialization and possibly pick a new input CU packet.
    for (uint32_t ch = 0; ch < num_ch_; ch++) {
      auto& link = cu_to_mc_link_[ch];
      if (!link.valid) {
        // Pick a CU input with a head packet destined to this channel.
        uint32_t start = rr_in_sel_cu_to_mc_[ch] % num_cus_;
        for (uint32_t k = 0; k < num_cus_; k++) {
          uint32_t cu = (start + k) % num_cus_;
          if (!cu_out[cu].can_pop_curr()) continue;
          const Packet& p = cu_out[cu].front_curr();
          if (p.dst_dram_chan != ch) continue;
          if (!noc_to_mc[ch].can_push_next()) break; // backpressure at destination
          auto popped = cu_out[cu].pop_curr();
          link.valid = true;
          link.pkt = std::move(*popped);
          link.remaining_latency = hop_lat_;
          link.remaining_bytes = link.pkt.bytes;
          rr_in_sel_cu_to_mc_[ch] = cu + 1;
          break;
        }
      }
 
      if (link.valid) {
        cu_to_mc_busy_cycles_++;
        if (link.remaining_latency > 0) {
          link.remaining_latency--;
        } else {
          uint32_t sent = std::min(bpc_, link.remaining_bytes);
          link.remaining_bytes -= sent;
          if (link.remaining_bytes == 0) {
            // Deliver at end of cycle (registered via push_next)
            bool ok = noc_to_mc[ch].push_next(std::move(link.pkt));
            (void)ok;
            assert(ok && "NoC deliver should respect noc_to_mc capacity check");
            link.valid = false;
          }
        }
      }
    }
  }
 
  void tick_dir_mc_to_cu(std::vector<RegFifo<Packet>>& mc_out, std::vector<RegFifo<Packet>>& noc_to_cu) {
    // For each CU output, advance serialization and possibly pick a new MC response.
    for (uint32_t cu = 0; cu < num_cus_; cu++) {
      auto& link = mc_to_cu_link_[cu];
      if (!link.valid) {
        // Pick a MC channel input with a head packet destined to this CU.
        uint32_t start = rr_in_sel_mc_to_cu_[cu] % num_ch_;
        for (uint32_t k = 0; k < num_ch_; k++) {
          uint32_t ch = (start + k) % num_ch_;
          if (!mc_out[ch].can_pop_curr()) continue;
          const Packet& p = mc_out[ch].front_curr();
          if (p.src_cu != cu) continue; // src_cu indicates destination CU for responses/acks
          if (!noc_to_cu[cu].can_push_next()) break;
          auto popped = mc_out[ch].pop_curr();
          link.valid = true;
          link.pkt = std::move(*popped);
          link.remaining_latency = hop_lat_;
          link.remaining_bytes = link.pkt.bytes;
          rr_in_sel_mc_to_cu_[cu] = ch + 1;
          break;
        }
      }
 
      if (link.valid) {
        mc_to_cu_busy_cycles_++;
        if (link.remaining_latency > 0) {
          link.remaining_latency--;
        } else {
          uint32_t sent = std::min(bpc_, link.remaining_bytes);
          link.remaining_bytes -= sent;
          if (link.remaining_bytes == 0) {
            bool ok = noc_to_cu[cu].push_next(std::move(link.pkt));
            (void)ok;
            assert(ok && "NoC deliver should respect noc_to_cu capacity check");
            link.valid = false;
          }
        }
      }
    }
  }
 
  uint32_t num_cus_ = 0;
  uint32_t num_ch_ = 0;
  uint32_t hop_lat_ = 0;
  uint32_t bpc_ = 0;
 
  std::vector<LinkInFlight> cu_to_mc_link_;
  std::vector<LinkInFlight> mc_to_cu_link_;
  std::vector<uint32_t> rr_in_sel_cu_to_mc_;
  std::vector<uint32_t> rr_in_sel_mc_to_cu_;
 
  uint64_t cu_to_mc_busy_cycles_ = 0;
  uint64_t mc_to_cu_busy_cycles_ = 0;
};
 
class DramChannel {
 public:
  struct PendingResp {
    uint64_t ready_cycle = 0;
    Packet pkt{};
    bool operator<(const PendingResp& other) const { return ready_cycle > other.ready_cycle; } // min-heap
  };
 
  DramChannel(uint32_t chan_id, uint32_t latency_cycles, uint32_t bytes_per_cycle)
      : chan_id_(chan_id), lat_(latency_cycles), bpc_(bytes_per_cycle) {}
 
  void tick(uint64_t cycle, RegFifo<Packet>* in_req, RegFifo<Packet>* out_rsp) {
    // Accept 0+ requests/cycle (limited by input FIFO and internal scheduling).
    // For cycle-accuracy with a single shared data bus, we serialize by tracking bus_free_cycle_.
    while (in_req->can_pop_curr()) {
      auto p_opt = in_req->pop_curr();
      if (!p_opt) break;
      Packet p = std::move(*p_opt);
      uint64_t transfer_cycles = std::max<uint64_t>(1, ceil_div_u64(p.bytes, bpc_));
      uint64_t start = std::max<uint64_t>(cycle, bus_free_cycle_);
      bus_free_cycle_ = start + transfer_cycles;
      uint64_t finish = start + lat_ + transfer_cycles;
 
      // Generate response/ack packet (bytes model serialization on return NoC too).
      Packet rsp;
      rsp.src_cu = p.src_cu;            // destination CU for return path
      rsp.dst_dram_chan = chan_id_;
      rsp.addr = p.addr;
      rsp.bytes = p.bytes;
      rsp.task_id = p.task_id;
      rsp.tag = p.tag;
      rsp.type = (p.type == PacketType::ReadReq) ? PacketType::ReadResp : PacketType::WriteAck;
      pending_.push(PendingResp{finish, std::move(rsp)});
      dram_busy_cycles_ += transfer_cycles;
    }
 
    // Emit ready responses if possible (registered output).
    while (!pending_.empty() && pending_.top().ready_cycle <= cycle) {
      if (!out_rsp->can_push_next()) break;
      out_rsp->push_next(std::move(pending_.top().pkt));
      pending_.pop();
    }
  }
 
  bool idle() const { return pending_.empty(); }
  uint64_t dram_busy_cycles() const { return dram_busy_cycles_; }
 
 private:
  uint32_t chan_id_ = 0;
  uint32_t lat_ = 0;
  uint32_t bpc_ = 0;
  uint64_t bus_free_cycle_ = 0;
 
  std::priority_queue<PendingResp> pending_;
  uint64_t dram_busy_cycles_ = 0;
};
 
class ComputeUnit {
 public:
  enum class State : uint8_t {
    Idle = 0,
    Gemm_Run = 1,
    Gemm_StoreWaitAck = 2,
    Softmax_LoadWait = 3,
    Softmax_Compute = 4,
    Softmax_StoreWaitAck = 5,
  };
 
  explicit ComputeUnit(uint32_t cu_id, const SimConfig* cfg) : id_(cu_id), cfg_(cfg) {}
 
  void tick(uint64_t cycle,
            RegFifo<TileTask>* task_in,
            RegFifo<Packet>* to_noc,
            RegFifo<Packet>* from_noc,
            TaskGraph* graph,
            Scheduler* scheduler,
            uint64_t* global_enqueue_counter) {
    // Drain incoming responses/acks.
    while (from_noc->can_pop_curr()) {
      auto p_opt = from_noc->pop_curr();
      if (!p_opt) break;
      on_packet(std::move(*p_opt));
    }
 
    // If idle, try to accept a new task.
    if (state_ == State::Idle) {
      if (task_in->can_pop_curr()) {
        auto t_opt = task_in->pop_curr();
        if (t_opt) {
          cur_ = std::move(*t_opt);
          cur_.enqueue_order = (*global_enqueue_counter)++;
          start_task(cycle);
        }
      }
    }
 
    // Advance active state.
    switch (state_) {
      case State::Idle:
        break;
      case State::Gemm_Run:
        tick_gemm(cycle, to_noc);
        break;
      case State::Gemm_StoreWaitAck:
        if (store_ack_) {
          on_task_complete(graph, scheduler);
        }
        break;
      case State::Softmax_LoadWait:
        // Waiting for read response. Injection is handled via try_inject_softmax_load().
        break;
      case State::Softmax_Compute:
        tick_softmax_compute(cycle, to_noc);
        break;
      case State::Softmax_StoreWaitAck:
        if (store_ack_) {
          on_task_complete(graph, scheduler);
        }
        break;
    }
 
    // Counters
    if (compute_busy_cycles_ > 0 && compute_remaining_ > 0) {
      (void)cycle;
    }
  }
 
  uint64_t completed_tasks() const { return completed_tasks_; }
  uint64_t compute_busy_cycles() const { return compute_busy_cycles_; }
  State state() const { return state_; }
 
 private:
  // Packet tags:
  // - For GEMM: tag = (phase << 16) | k_step (phase: 0=A, 1=B, 2=Cwrite)
  // - For Softmax: tag = (phase << 16) | 0 (phase: 10=Load, 11=Store)
  static constexpr uint32_t TAG_PHASE_SHIFT = 16;
  static constexpr uint32_t PHASE_GEMM_A = 0;
  static constexpr uint32_t PHASE_GEMM_B = 1;
  static constexpr uint32_t PHASE_GEMM_C = 2;
  static constexpr uint32_t PHASE_SMX_LOAD = 10;
  static constexpr uint32_t PHASE_SMX_STORE = 11;
 
  uint32_t map_addr_to_chan(uint64_t addr) const {
    // Simple interleaving by cacheline-ish granularity.
    constexpr uint64_t line = 128;
    return static_cast<uint32_t>((addr / line) % cfg_->num_dram_channels);
  }
 
  void start_task(uint64_t /*cycle*/) {
    // Reset per-task state.
    have_a_.clear();
    have_b_.clear();
    gemm_k_steps_issued_a_.clear();
    gemm_k_steps_issued_b_.clear();
    a_base_ = cur_.a_base;
    b_base_ = cur_.b_base;
    c_base_ = cur_.c_base;
    store_ack_ = false;
    load_done_ = false;
    compute_remaining_ = 0;
    softmax_phase_ = 0;
    load_injected_ = false;
    store_issued_ = false;
 
    if (cur_.op == OpType::Gemm) {
      gemm_k_total_ = cur_.gemm.k_total;
      gemm_k_step_ = cur_.gemm.kt;
      gemm_k_idx_ = 0;
      // Number of K-tiles.
      gemm_num_k_tiles_ = static_cast<uint32_t>(ceil_div_u64(gemm_k_total_, gemm_k_step_));
      have_a_.assign(gemm_num_k_tiles_, false);
      have_b_.assign(gemm_num_k_tiles_, false);
      gemm_k_steps_issued_a_.assign(gemm_num_k_tiles_, false);
      gemm_k_steps_issued_b_.assign(gemm_num_k_tiles_, false);
      state_ = State::Gemm_Run;
    } else {
      // Softmax starts with load.
      issue_softmax_load();
      state_ = State::Softmax_LoadWait;
    }
  }
 
  void on_task_complete(TaskGraph* graph, Scheduler* scheduler) {
    completed_tasks_++;
    auto newly_ready = graph->mark_done(cur_.id);
    scheduler->on_newly_ready(newly_ready);
    state_ = State::Idle;
  }
 
  void on_packet(Packet&& p) {
    // For operator-level modeling we treat the response bytes as "operand now available".
    uint32_t phase = (p.tag >> TAG_PHASE_SHIFT);
    uint32_t idx = (p.tag & 0xFFFFu);
 
    if (p.task_id != cur_.id) {
      // In this initial model, tasks are serialized per CU, so any stray response is an error.
      // (If you later overlap tasks, you'll route by task_id.)
      return;
    }
 
    if (p.type == PacketType::ReadResp) {
      if (phase == PHASE_GEMM_A && idx < have_a_.size()) {
        have_a_[idx] = true;
      } else if (phase == PHASE_GEMM_B && idx < have_b_.size()) {
        have_b_[idx] = true;
      } else if (phase == PHASE_SMX_LOAD) {
        load_done_ = true;
        // After load completes, schedule compute.
        setup_softmax_compute();
        state_ = State::Softmax_Compute;
      }
    } else if (p.type == PacketType::WriteAck) {
      if (phase == PHASE_GEMM_C) {
        store_ack_ = true;
      } else if (phase == PHASE_SMX_STORE) {
        store_ack_ = true;
      }
    }
  }
 
  void issue_mem_read(RegFifo<Packet>* to_noc, uint64_t addr, uint32_t bytes, uint32_t phase, uint32_t idx) {
    Packet p;
    p.type = PacketType::ReadReq;
    p.src_cu = id_;
    p.addr = addr;
    p.bytes = bytes;
    p.dst_dram_chan = map_addr_to_chan(addr);
    p.task_id = cur_.id;
    p.tag = (phase << TAG_PHASE_SHIFT) | idx;
    if (to_noc->can_push_next()) {
      bool ok = to_noc->push_next(std::move(p));
      (void)ok;
      assert(ok);
    }
  }
 
  void issue_mem_write(RegFifo<Packet>* to_noc, uint64_t addr, uint32_t bytes, uint32_t phase, uint32_t idx) {
    Packet p;
    p.type = PacketType::WriteReq;
    p.src_cu = id_;
    p.addr = addr;
    p.bytes = bytes;
    p.dst_dram_chan = map_addr_to_chan(addr);
    p.task_id = cur_.id;
    p.tag = (phase << TAG_PHASE_SHIFT) | idx;
    if (to_noc->can_push_next()) {
      bool ok = to_noc->push_next(std::move(p));
      (void)ok;
      assert(ok);
    }
  }
 
  void tick_gemm(uint64_t /*cycle*/, RegFifo<Packet>* to_noc) {
    // If currently computing, advance pipeline.
    if (compute_remaining_ > 0) {
      compute_remaining_--;
      compute_busy_cycles_++;
      if (compute_remaining_ == 0) {
        // Finished compute for this k-tile.
        gemm_k_idx_++;
      }
    }
 
    // If all K tiles computed, issue store once.
    if (gemm_k_idx_ >= gemm_num_k_tiles_) {
      if (!store_issued_) {
        uint32_t m = cur_.gemm.mt;
        uint32_t n = cur_.gemm.nt;
        uint32_t c_bytes = m * n * cfg_->gemm_elem_bytes;
        uint64_t addr = c_base_ + static_cast<uint64_t>(cur_.gemm.m0) * 4096ull + cur_.gemm.n0 * 64ull;
        issue_mem_write(to_noc, addr, c_bytes, PHASE_GEMM_C, 0);
        store_issued_ = true;
        state_ = State::Gemm_StoreWaitAck;
      }
      return;
    }
 
    // Prefetch operands for current k tile if not present.
    uint32_t k_tile = gemm_k_idx_;
    if (!have_a_[k_tile] && !gemm_k_steps_issued_a_[k_tile]) {
      uint32_t m = cur_.gemm.mt;
      uint32_t k = cur_.gemm.kt;
      uint32_t a_bytes = m * k * cfg_->gemm_elem_bytes;
      uint64_t addr = a_base_ + static_cast<uint64_t>(cur_.gemm.m0) * 4096ull + (cur_.gemm.k_total ? k_tile * 64ull : 0ull);
      if (to_noc->can_push_next()) {
        issue_mem_read(to_noc, addr, a_bytes, PHASE_GEMM_A, k_tile);
        gemm_k_steps_issued_a_[k_tile] = true;
      }
    }
    if (!have_b_[k_tile] && !gemm_k_steps_issued_b_[k_tile]) {
      uint32_t n = cur_.gemm.nt;
      uint32_t k = cur_.gemm.kt;
      uint32_t b_bytes = k * n * cfg_->gemm_elem_bytes;
      uint64_t addr = b_base_ + static_cast<uint64_t>(cur_.gemm.n0) * 4096ull + (cur_.gemm.k_total ? k_tile * 64ull : 0ull);
      if (to_noc->can_push_next()) {
        issue_mem_read(to_noc, addr, b_bytes, PHASE_GEMM_B, k_tile);
        gemm_k_steps_issued_b_[k_tile] = true;
      }
    }
 
    // Start compute for current k tile if operands ready and compute engine idle.
    if (compute_remaining_ == 0 && have_a_[k_tile] && have_b_[k_tile]) {
      uint64_t flops = 2ull * cur_.gemm.mt * cur_.gemm.nt * cur_.gemm.kt;
      compute_remaining_ = std::max<uint64_t>(1, ceil_div_u64(flops, cfg_->cu_flops_per_cycle));
      // Note: compute will begin consuming cycles starting next tick of this function;
      // we model it as "issued" here and count cycles as we decrement.
    }
  }
 
  void issue_softmax_load() {
    // Load a contiguous row tile.
    store_issued_ = false;
    store_ack_ = false;
    load_done_ = false;
    uint32_t bytes = cur_.softmax.lt * cfg_->softmax_elem_bytes;
    uint64_t addr = a_base_ + static_cast<uint64_t>(cur_.softmax.row) * 8192ull;
    pending_softmax_load_ = Packet{};
    pending_softmax_load_.type = PacketType::ReadReq;
    pending_softmax_load_.src_cu = id_;
    pending_softmax_load_.addr = addr;
    pending_softmax_load_.bytes = bytes;
    pending_softmax_load_.dst_dram_chan = map_addr_to_chan(addr);
    pending_softmax_load_.task_id = cur_.id;
    pending_softmax_load_.tag = (PHASE_SMX_LOAD << TAG_PHASE_SHIFT) | 0;
    softmax_phase_ = 0;
  }
 
  void setup_softmax_compute() {
    // Compute cycles for: reduce-max + exp+sum + normalize.
    // This is operator-level but cycle-accurate wrt configured vector/SFU throughput.
    uint64_t L = cur_.softmax.lt;
    uint64_t reduce_cycles = ceil_div_u64(L, cfg_->cu_vec_elems_per_cycle); // max reduction pass
    uint64_t exp_cycles = ceil_div_u64(L, cfg_->cu_sfu_elems_per_cycle);    // exp throughput
    uint64_t sum_cycles = ceil_div_u64(L, cfg_->cu_vec_elems_per_cycle);    // sum reduction pass
    uint64_t norm_cycles = ceil_div_u64(L, cfg_->cu_vec_elems_per_cycle);   // normalization pass
    compute_remaining_ = std::max<uint64_t>(1, reduce_cycles + exp_cycles + sum_cycles + norm_cycles);
    softmax_phase_ = 1;
  }
 
  void tick_softmax_compute(uint64_t /*cycle*/, RegFifo<Packet>* to_noc) {
    if (compute_remaining_ > 0) {
      compute_remaining_--;
      compute_busy_cycles_++;
    }
 
    if (compute_remaining_ == 0 && softmax_phase_ == 1) {
      // Issue store.
      if (!store_issued_) {
        uint32_t bytes = cur_.softmax.lt * cfg_->softmax_elem_bytes;
        uint64_t addr = c_base_ + static_cast<uint64_t>(cur_.softmax.row) * 8192ull;
        issue_mem_write(to_noc, addr, bytes, PHASE_SMX_STORE, 0);
        store_issued_ = true;
        state_ = State::Softmax_StoreWaitAck;
      }
    }
  }
 
 public:
  // Set by top-level sim so CU can mark task completion in the global graph.
  void bind_graph(TaskGraph* g, Scheduler* s) {
    graph_ = g;
    scheduler_ = s;
  }
 
  // Softmax load injection is stored as a packet so CU tick can inject it respecting backpressure.
  void try_inject_softmax_load(RegFifo<Packet>* to_noc) {
    if (state_ == State::Softmax_LoadWait && !load_injected_) {
      if (to_noc->can_push_next()) {
        to_noc->push_next(pending_softmax_load_);
        load_injected_ = true;
      }
    }
  }
 
 private:
  uint32_t id_ = 0;
  const SimConfig* cfg_ = nullptr;
 
  State state_ = State::Idle;
  TileTask cur_{};
 
  // GEMM state
  uint32_t gemm_k_total_ = 0;
  uint32_t gemm_k_step_ = 0;
  uint32_t gemm_num_k_tiles_ = 0;
  uint32_t gemm_k_idx_ = 0;
  std::vector<bool> have_a_;
  std::vector<bool> have_b_;
  std::vector<bool> gemm_k_steps_issued_a_;
  std::vector<bool> gemm_k_steps_issued_b_;
 
  // Softmax state
  uint32_t softmax_phase_ = 0;
  Packet pending_softmax_load_{};
  bool load_injected_ = false;
  bool load_done_ = false;
 
  // Common
  uint64_t a_base_ = 0, b_base_ = 0, c_base_ = 0;
  uint64_t compute_remaining_ = 0;
  bool store_issued_ = false;
  bool store_ack_ = false;
 
  TaskGraph* graph_ = nullptr;
  Scheduler* scheduler_ = nullptr;
 
  uint64_t completed_tasks_ = 0;
  uint64_t compute_busy_cycles_ = 0;
};
 
struct CliArgs {
  // Workload defaults
  uint32_t gemm_m = 2048, gemm_n = 2048, gemm_k = 2048;
  uint32_t gemm_mt = 128, gemm_nt = 128, gemm_kt = 32;
 
  uint32_t softmax_rows = 1024;
  uint32_t softmax_cols = 4096;
  uint32_t softmax_lt = 1024;
 
  SimConfig cfg{};
};
 
static inline bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}
 
static inline uint64_t parse_u64(const std::string& s) {
  return static_cast<uint64_t>(std::stoull(s));
}
 
static CliArgs parse_args(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto eat = [&](const std::string& key) -> std::optional<std::string> {
      if (arg == key && i + 1 < argc) return std::string(argv[++i]);
      if (starts_with(arg, key + "=")) return arg.substr(key.size() + 1);
      return std::nullopt;
    };
 
    if (auto v = eat("--num_cus")) a.cfg.num_cus = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--num_dram_channels")) a.cfg.num_dram_channels = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--max_cycles")) a.cfg.max_cycles = static_cast<uint32_t>(parse_u64(*v));
 
    else if (auto v = eat("--noc_latency")) a.cfg.noc_hop_latency_cycles = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--noc_bpc")) a.cfg.noc_bytes_per_cycle = static_cast<uint32_t>(parse_u64(*v));
 
    else if (auto v = eat("--dram_latency")) a.cfg.dram_latency_cycles = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--dram_bpc")) a.cfg.dram_bytes_per_cycle = static_cast<uint32_t>(parse_u64(*v));
 
    else if (auto v = eat("--cu_flops_per_cycle")) a.cfg.cu_flops_per_cycle = parse_u64(*v);
    else if (auto v = eat("--cu_vec_epc")) a.cfg.cu_vec_elems_per_cycle = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--cu_sfu_epc")) a.cfg.cu_sfu_elems_per_cycle = static_cast<uint32_t>(parse_u64(*v));
 
    else if (auto v = eat("--gemm_m")) a.gemm_m = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--gemm_n")) a.gemm_n = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--gemm_k")) a.gemm_k = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--gemm_mt")) a.gemm_mt = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--gemm_nt")) a.gemm_nt = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--gemm_kt")) a.gemm_kt = static_cast<uint32_t>(parse_u64(*v));
 
    else if (auto v = eat("--softmax_rows")) a.softmax_rows = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--softmax_cols")) a.softmax_cols = static_cast<uint32_t>(parse_u64(*v));
    else if (auto v = eat("--softmax_lt")) a.softmax_lt = static_cast<uint32_t>(parse_u64(*v));
  }
  return a;
}
 
static Workload build_workload(const CliArgs& args) {
  Workload w;
  uint64_t id = 1;
  uint64_t enqueue_order = 0;
 
  // Synthetic base addresses for tensors (just unique regions).
  uint64_t A = 0x10000000ull;
  uint64_t B = 0x20000000ull;
  uint64_t C = 0x30000000ull;
  uint64_t X = 0x40000000ull;
  uint64_t Y = 0x50000000ull;
 
  // GEMM tiles for a single GEMM op.
  for (uint32_t m0 = 0; m0 < args.gemm_m; m0 += args.gemm_mt) {
    for (uint32_t n0 = 0; n0 < args.gemm_n; n0 += args.gemm_nt) {
      TileTask t;
      t.id = id++;
      t.op = OpType::Gemm;
      t.enqueue_order = enqueue_order++;
      t.a_base = A;
      t.b_base = B;
      t.c_base = C;
      t.gemm.mt = std::min(args.gemm_mt, args.gemm_m - m0);
      t.gemm.nt = std::min(args.gemm_nt, args.gemm_n - n0);
      t.gemm.kt = args.gemm_kt;
      t.gemm.m0 = m0;
      t.gemm.n0 = n0;
      t.gemm.k_total = args.gemm_k;
      w.tasks.push_back(std::move(t));
    }
  }
 
  // Softmax tiles for a single softmax op (row-wise over cols).
  // Initial model: one task per row-tile segment; no cross-tile reductions.
  for (uint32_t r = 0; r < args.softmax_rows; r++) {
    for (uint32_t c0 = 0; c0 < args.softmax_cols; c0 += args.softmax_lt) {
      TileTask t;
      t.id = id++;
      t.op = OpType::Softmax;
      t.enqueue_order = enqueue_order++;
      t.a_base = X + static_cast<uint64_t>(c0) * args.cfg.softmax_elem_bytes;
      t.c_base = Y + static_cast<uint64_t>(c0) * args.cfg.softmax_elem_bytes;
      t.softmax.row = r;
      t.softmax.lt = std::min(args.softmax_lt, args.softmax_cols - c0);
      t.softmax.l_total = args.softmax_cols;
      w.tasks.push_back(std::move(t));
    }
  }
 
  return w;
}
 
struct SimState {
  uint64_t cycle = 0;
};
 
static int run_sim(const CliArgs& args) {
  const SimConfig& cfg = args.cfg;
  Workload w = build_workload(args);
  TaskGraph graph(w.tasks);
 
  // Per-CU task inputs (from scheduler).
  std::vector<RegFifo<TileTask>> cu_task_in(cfg.num_cus);
  for (auto& q : cu_task_in) q.set_capacity(256);
 
  Scheduler sched(cfg.num_cus, &graph, &cu_task_in);
  sched.seed_initial_ready(w.tasks);
 
  // NoC connection FIFOs.
  std::vector<RegFifo<Packet>> cu_to_noc(cfg.num_cus);
  std::vector<RegFifo<Packet>> noc_to_cu(cfg.num_cus);
  for (auto& q : cu_to_noc) q.set_capacity(cfg.noc_injection_fifo_depth);
  for (auto& q : noc_to_cu) q.set_capacity(cfg.noc_ejection_fifo_depth);
 
  std::vector<RegFifo<Packet>> noc_to_mc(cfg.num_dram_channels);
  std::vector<RegFifo<Packet>> mc_to_noc(cfg.num_dram_channels);
  for (auto& q : noc_to_mc) q.set_capacity(cfg.dram_req_fifo_depth);
  for (auto& q : mc_to_noc) q.set_capacity(cfg.dram_rsp_fifo_depth);
 
  NoC noc(cfg.num_cus, cfg.num_dram_channels, cfg.noc_hop_latency_cycles, cfg.noc_bytes_per_cycle);
 
  std::vector<DramChannel> dram;
  dram.reserve(cfg.num_dram_channels);
  for (uint32_t ch = 0; ch < cfg.num_dram_channels; ch++) {
    dram.emplace_back(ch, cfg.dram_latency_cycles, cfg.dram_bytes_per_cycle);
  }
 
  std::vector<ComputeUnit> cus;
  cus.reserve(cfg.num_cus);
  for (uint32_t cu = 0; cu < cfg.num_cus; cu++) {
    cus.emplace_back(cu, &cfg);
  }
  for (auto& cu : cus) cu.bind_graph(&graph, &sched);
 
  SimState st;
  uint64_t global_enqueue_counter = 0;
 
  auto commit_all = [&]() {
    for (auto& q : cu_task_in) q.commit();
    for (auto& q : cu_to_noc) q.commit();
    for (auto& q : noc_to_cu) q.commit();
    for (auto& q : noc_to_mc) q.commit();
    for (auto& q : mc_to_noc) q.commit();
  };
 
  // Initial dispatch happens at cycle 0 into incoming queues; visible at cycle 1 after commit.
  sched.tick_dispatch();
  commit_all();
 
  for (st.cycle = 0; st.cycle < cfg.max_cycles; st.cycle++) {
    // 1) Dispatch ready tasks into CU input FIFOs (registered).
    sched.tick_dispatch();
 
    // 2) CU ticks: consume responses, advance task FSMs, enqueue mem reqs.
    for (uint32_t cu = 0; cu < cfg.num_cus; cu++) {
      // Allow softmax load injection respecting registered behavior.
      cus[cu].try_inject_softmax_load(&cu_to_noc[cu]);
      cus[cu].tick(st.cycle, &cu_task_in[cu], &cu_to_noc[cu], &noc_to_cu[cu], &graph, &sched, &global_enqueue_counter);
    }
 
    // 3) NoC transfers packets in both directions.
    noc.tick(&cu_to_noc, &noc_to_mc, &mc_to_noc, &noc_to_cu);
 
    // 4) DRAM channels accept requests and schedule responses.
    for (uint32_t ch = 0; ch < cfg.num_dram_channels; ch++) {
      dram[ch].tick(st.cycle, &noc_to_mc[ch], &mc_to_noc[ch]);
    }
 
    // 5) Commit registered queues (everything produced this cycle becomes visible next cycle).
    commit_all();
 
    // 6) Stop condition: all tasks done and no in-flight traffic.
    bool fifos_empty = true;
    for (auto& q : cu_task_in) fifos_empty &= q.empty_all();
    for (auto& q : cu_to_noc) fifos_empty &= q.empty_all();
    for (auto& q : noc_to_cu) fifos_empty &= q.empty_all();
    for (auto& q : noc_to_mc) fifos_empty &= q.empty_all();
    for (auto& q : mc_to_noc) fifos_empty &= q.empty_all();
    bool dram_idle = true;
    for (auto& d : dram) dram_idle &= d.idle();
 
    if (graph.is_done() && fifos_empty && dram_idle && noc.idle()) {
      break;
    }
  }
 
  // Report.
  std::cout << "accel_sim results\n";
  std::cout << "  cycles: " << st.cycle << "\n";
  std::cout << "  tasks:  " << graph.completed_tasks() << " / " << graph.total_tasks() << "\n";
 
  uint64_t total_compute_busy = 0;
  for (uint32_t cu = 0; cu < cfg.num_cus; cu++) {
    total_compute_busy += cus[cu].compute_busy_cycles();
    std::cout << "  CU[" << cu << "]: completed=" << cus[cu].completed_tasks()
              << " compute_busy_cycles=" << cus[cu].compute_busy_cycles() << "\n";
  }
  std::cout << "  NoC busy cycles (aggregate over links):\n";
  std::cout << "    CU->MC: " << noc.cu_to_mc_busy_cycles() << "\n";
  std::cout << "    MC->CU: " << noc.mc_to_cu_busy_cycles() << "\n";
 
  uint64_t dram_busy = 0;
  for (uint32_t ch = 0; ch < cfg.num_dram_channels; ch++) dram_busy += dram[ch].dram_busy_cycles();
  std::cout << "  DRAM busy cycles (aggregate over channels): " << dram_busy << "\n";
 
  // Simple utilization-ish hints.
  if (st.cycle > 0) {
    double cu_util = static_cast<double>(total_compute_busy) / (static_cast<double>(st.cycle) * cfg.num_cus);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  approx CU compute utilization: " << (100.0 * cu_util) << "%\n";
  }
 
  return 0;
}
 
} // namespace sim
 
int main(int argc, char** argv) {
  auto args = sim::parse_args(argc, argv);
  return sim::run_sim(args);
}


