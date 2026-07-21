/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "InsertWaitAluPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/RegHalfKeyer.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace {
using namespace stinkytofu;

// ---------------------------------------------------------------------------
// Mode 2 counters and events (VA_VDST, VM_VSRC).
// ---------------------------------------------------------------------------

enum CounterType : uint8_t {
    CT_VA_VDST = 0,
    CT_VM_VSRC = 1,
    NUM_COUNTERS = 2,
};

enum WaitEventType : uint8_t {
    // VA_VDST events: VALU VGPR-dest writes.
    EV_VGPR_CSMACC_WRITE = 0,  // core/side-MACC VALU (v_add_f32, v_mul_f32, v_mfma, ...)
    EV_VGPR_DPMACC_WRITE,      // double-precision MACC (v_add/mul/fma_f64, f64 cmp, v_cvt_u32_f64)
    EV_VGPR_TRANS_WRITE,       // transcendental VALU, 32- and 64-bit (v_rcp_f32, v_rcp_f64, ...)
    EV_VGPR_XDL_WRITE,         // XDL WMMA / SWMMAC
    // VM_VSRC events
    EV_VGPR_LDS_READ,   // ds_read / ds_write reading a VGPR source
    EV_VGPR_FLAT_READ,  // FLAT reading a VGPR source
    EV_VGPR_VMEM_READ,  // buffer / global / image reading a VGPR source
    EV_NUM,
};

// Compact bitset over WaitEventType; twoOrMore() flags "out-of-order" completion
// (more than one event class pending on a single counter ⇒ force wait(0)).
struct WaitEventSet {
    uint32_t mask = 0;
    void insert(WaitEventType e) {
        mask |= 1u << e;
    }
    bool contains(WaitEventType e) const {
        return mask & (1u << e);
    }
    bool containsAll(WaitEventSet o) const {
        return (mask & o.mask) == o.mask;
    }
    bool twoOrMore() const {
        return (mask & (mask - 1)) != 0;
    }
    WaitEventSet operator|(WaitEventSet o) const {
        return {mask | o.mask};
    }
    WaitEventSet operator&(WaitEventSet o) const {
        return {mask & o.mask};
    }
    WaitEventSet operator~() const {
        return {~mask};
    }
    bool operator==(WaitEventSet o) const {
        return mask == o.mask;
    }
};

inline CounterType counterFromEvent(WaitEventType e) {
    switch (e) {
        case EV_VGPR_CSMACC_WRITE:
        case EV_VGPR_DPMACC_WRITE:
        case EV_VGPR_TRANS_WRITE:
        case EV_VGPR_XDL_WRITE:
            return CT_VA_VDST;
        default:
            return CT_VM_VSRC;
    }
}

// Per-pipe lanes. Each hardware counter aggregates several sub-pipelines that
// complete in issue order internally yet out-of-order with one another:
//   VA_VDST → CSMACC, DPMACC, TRANS, XDL
//   VM_VSRC → LDS, FLAT, VMEM
// Tracking each pipe separately lets us emit the tightest safe wait: to prove a
// producer in pipe E is done, we only need enough same-pipe followers to have
// drained, regardless of the other pipes' depth. The lane order mirrors
// WaitEventType, so laneOfEvent is a straight cast.
enum Lane : uint8_t {
    LANE_CSMACC = 0,
    LANE_DPMACC = 1,
    LANE_TRANS = 2,
    LANE_XDL = 3,
    LANE_LDS = 4,
    LANE_FLAT = 5,
    LANE_VMEM = 6,
    NUM_LANES = 7,
};

inline Lane laneOfEvent(WaitEventType e) {
    return static_cast<Lane>(e);  // WaitEventType and Lane share ordering
}

// VM_VSRC ordering groups (hardware FIFOs). Unlike the VA_VDST sub-pipes, the
// VM_VSRC memory classes decrement in-order only *within* an ordering group:
//   Group A (LDS FIFO):  ds_*  and flat_*
//   Group B (TEX FIFO):  buffer/global/scratch/image  and flat_*
// flat_* is a member of BOTH groups (hardware places it in both FIFOs), so a
// flat producer accrues followers in either group. Tracking followers per group
// lets us emit a precise vm_vsrc(f) instead of a full drain whenever f>0 — other
// groups only inflate the shared counter, never deflate it, so cross-group mixes
// are safe. The only drain case is f==0 (producer is last of its group).
enum VMGroup : uint8_t {
    VM_GROUP_A = 0,  // LDS FIFO
    VM_GROUP_B = 1,  // TEX FIFO
    NUM_VM_GROUPS = 2,
};

inline bool vmInGroupA(WaitEventType e) {
    return e == EV_VGPR_LDS_READ || e == EV_VGPR_FLAT_READ;
}
inline bool vmInGroupB(WaitEventType e) {
    return e == EV_VGPR_VMEM_READ || e == EV_VGPR_FLAT_READ;
}

inline const char* laneName(Lane l) {
    switch (l) {
        case LANE_CSMACC:
            return "CSMACC";
        case LANE_DPMACC:
            return "DPMACC";
        case LANE_TRANS:
            return "TRANS";
        case LANE_XDL:
            return "XDL";
        case LANE_LDS:
            return "LDS";
        case LANE_FLAT:
            return "FLAT";
        default:
            return "VMEM";
    }
}

// Lane index range [lo, hi) owned by a counter.
inline int counterLaneLo(CounterType c) {
    return c == CT_VA_VDST ? LANE_CSMACC : LANE_LDS;
}
inline int counterLaneHi(CounterType c) {
    return c == CT_VA_VDST ? LANE_LDS : NUM_LANES;
}

inline const char* counterName(CounterType c) {
    return c == CT_VA_VDST ? "va_vdst" : "vm_vsrc";
}

inline const char* eventName(WaitEventType e) {
    switch (e) {
        case EV_VGPR_CSMACC_WRITE:
            return "CSMACC_WRITE";
        case EV_VGPR_DPMACC_WRITE:
            return "DPMACC_WRITE";
        case EV_VGPR_TRANS_WRITE:
            return "TRANS_WRITE";
        case EV_VGPR_XDL_WRITE:
            return "XDL_WRITE";
        case EV_VGPR_LDS_READ:
            return "LDS_READ";
        case EV_VGPR_FLAT_READ:
            return "FLAT_READ";
        case EV_VGPR_VMEM_READ:
            return "VMEM_READ";
        default:
            return "?";
    }
}

inline WaitEventSet eventsForCounter(CounterType c) {
    WaitEventSet s;
    if (c == CT_VA_VDST) {
        s.insert(EV_VGPR_CSMACC_WRITE);
        s.insert(EV_VGPR_DPMACC_WRITE);
        s.insert(EV_VGPR_TRANS_WRITE);
        s.insert(EV_VGPR_XDL_WRITE);
    } else {
        s.insert(EV_VGPR_LDS_READ);
        s.insert(EV_VGPR_FLAT_READ);
        s.insert(EV_VGPR_VMEM_READ);
    }
    return s;
}

// Comma-separated list of pending event names in `ev`, e.g. "XDL_WRITE,CSMACC_WRITE".
// Used by the wait-hit debug print when ooo=1 fires, to make it clear *which*
// event classes are causing the conservative full-drain.
inline std::string pendingEventsStr(WaitEventSet ev) {
    std::string out;
    for (int i = 0; i < EV_NUM; ++i) {
        auto e = static_cast<WaitEventType>(i);
        if (ev.contains(e)) {
            if (!out.empty()) out += ",";
            out += eventName(e);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Instruction classifiers
// ---------------------------------------------------------------------------

// Map an instruction to its (single) mode2 event class, or none if it is
// neither a VALU producer nor a VMEM/LDS/FLAT consumer.
// VALU completion-class order: XDL -> TRANS -> DPMACC -> CSMACC. f64
// transcendentals carry both the TRANS and DPMACC properties; TRANS is matched
// first so they classify as TRANS.
std::optional<WaitEventType> classifyEvent(const StinkyInstruction& inst) {
    if (isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst)) {
        if (isXDLWMMA(inst)) return EV_VGPR_XDL_WRITE;
        if (isTranscendental(inst)) return EV_VGPR_TRANS_WRITE;  // 32- and 64-bit
        if (isDPMACC(inst)) return EV_VGPR_DPMACC_WRITE;
        return EV_VGPR_CSMACC_WRITE;
    }
    if (isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst)) return EV_VGPR_LDS_READ;
    if (isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst)) return EV_VGPR_FLAT_READ;
    // VMEM family. Stinkytofu does not yet flag scratch / image / sample / BVH
    // instructions; on archs that emit them they belong in this same bucket.
    if (isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) || isGLOBALOrAtomic(inst))
        return EV_VGPR_VMEM_READ;
    return std::nullopt;
}

// VOP3PX2 / VOP3PX3 are software-only encodings of a back-to-back VOP3P pair
// (LD_SCALE + WMMA). Hardware decodes each as two separate VOP3P sub-issues,
// both bumping VA_VDST, so software must count 2.
inline bool hasMatrixScalePair(const StinkyInstruction& inst) {
    auto mc = inst.getHwInstDesc()->microcode;
    return mc == MicrocodeFormat::MC_VOP3PX2 || mc == MicrocodeFormat::MC_VOP3PX3;
}

// Walk `regs`, skipping non-VGPR ones, and invoke fn(vgprIdx, half) for each
// VGPR. halfFn maps an operand's position (VGPR-only) to its True16 half
// selector, so fn may act at half-word (LOW/HIGH) granularity. Callers pass a
// single operand list (getSrcRegs or getDestRegs) — never both, since src and
// dst use different half selectors. Shared by producer stamping and consumer
// probing.
template <typename HalfFn, typename Fn>
inline void forEachVGPR(const std::vector<StinkyRegister>& regs, HalfFn&& halfFn, Fn&& fn) {
    size_t opIdx = 0;
    for (const auto& reg : regs) {
        if (reg.dataType != StinkyRegister::Type::Register) continue;
        if (reg.reg.type != RegType::V) continue;
        HighBitSel half = halfFn(opIdx);
        ++opIdx;
        for (uint16_t off = 0; off < reg.reg.num; ++off) fn(reg.reg.idx + off, half);
    }
}

// EXEC writes invalidate any non-zero VA_VDST wait (skipped VALUs don't bump
// the HW counter). Covers explicit destination and implicit destination via
// HW flag.
inline bool writesExec(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitWriteEXEC)) return true;
    for (const auto& d : inst.getDestRegs()) {
        if (d.dataType != StinkyRegister::Type::Register) continue;
        RegType t = d.reg.type;
        if (t == RegType::EXEC || t == RegType::EXEC_LO || t == RegType::EXEC_HI) return true;
    }
    return false;
}

inline bool isWaitAluInst(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_wait_alu;
}

// ---------------------------------------------------------------------------
// True16 half-selectors
// ---------------------------------------------------------------------------

// True16 half-selector for dest operand index `destIdx` (only operand 0 and 1
// can have a True16 dst-half). Falls through to NONE without modifier.
inline HighBitSel destHalfSel(const True16Modifiers* mod, size_t destIdx) {
    if (!mod) return HighBitSel::NONE;
    if (destIdx == 0) return mod->getDst0();
    if (destIdx == 1) return mod->getDst1();
    return HighBitSel::NONE;
}

inline HighBitSel srcHalfSel(const True16Modifiers* mod, size_t srcIdx) {
    return mod ? mod->getSrc(srcIdx) : HighBitSel::NONE;
}

// ---------------------------------------------------------------------------
// Wait struct
// ---------------------------------------------------------------------------

// Sentinel: "don't emit this field" for a per-counter wait value.
constexpr unsigned kNoWait = ~0u;

struct Wait {
    std::array<unsigned, NUM_COUNTERS> counts = {kNoWait, kNoWait};
    unsigned get(CounterType c) const {
        return counts[c];
    }
    void set(CounterType c, unsigned v) {
        counts[c] = v;
    }
    bool hasAny() const {
        return counts[CT_VA_VDST] != kNoWait || counts[CT_VM_VSRC] != kNoWait;
    }
};

inline void setNoWait(Wait& w, CounterType c) {
    w.set(c, kNoWait);
}
inline bool isNoWait(const Wait& w, CounterType c) {
    return w.get(c) == kNoWait;
}

inline void addWait(Wait& w, CounterType c, unsigned v) {
    w.set(c, std::min(w.get(c), v));
}

// SWaitAluData field widths: va_vdst is 4 bits, vm_vsrc is 3 bits. The all-ones
// value of each field is reserved as the "no-wait" sentinel, so the largest
// emittable real wait is (1 << width) - 2.
inline unsigned encodingSentinel(CounterType c) {
    return c == CT_VA_VDST ? 15u : 7u;
}
inline unsigned maxEmittableWait(CounterType c) {
    return encodingSentinel(c) - 1;
}

// ---------------------------------------------------------------------------
// WaitcntBrackets — UB/LB scoreboard with per-VGPR per-counter scores
// ---------------------------------------------------------------------------

// Per-VGPR producer stamp: the (lane, ordinal) of the last VA producer that
// wrote the reg, and of the last VM read that sourced it. ordinal==0 means "no
// producer on that side". The lane records which sub-pipeline produced it, so a
// consumer can count same-pipe followers.
struct VgprStamp {
    Lane vaLane = LANE_CSMACC;  // meaningful only when vaOrdinal != 0
    unsigned vaOrdinal = 0;     // cumulative position within vaLane
    Lane vmLane = LANE_LDS;     // meaningful only when vmOrdinal != 0
    unsigned vmOrdinal = 0;     // cumulative position within vmLane
    // VM_VSRC ordering-group ordinals. A pure-LDS producer sets only vmOrdA, a
    // pure-VMEM producer only vmOrdB; a flat_* producer sets BOTH (it lives in
    // both FIFOs). 0 means "not a member of that group".
    unsigned vmOrdA = 0;  // cumulative position within Group A (LDS FIFO)
    unsigned vmOrdB = 0;  // cumulative position within Group B (TEX FIFO)
};

class WaitcntBrackets {
   public:
    // Aggregate views of a counter (sum of its lanes). getScoreRange feeds the
    // EXEC-guard "any VALU in flight" test; the LB/UB variants are debug-only.
    unsigned getScoreLB(CounterType c) const {
        return laneSum(c, floor_);
    }
    unsigned getScoreUB(CounterType c) const {
        return laneSum(c, issued);
    }
    unsigned getScoreRange(CounterType c) const {
        return getScoreUB(c) - getScoreLB(c);
    }
    size_t scoresSize() const {
        return scores.size();
    }

    // Stamp scoreboard after instruction `inst` issues with event `ev`.
    // VA_VDST stamps each VGPR def+src, VM_VSRC stamps each VGPR src.
    void onProducer(WaitEventType ev, const StinkyInstruction& inst, const VGPRHalfKeyer& keyer) {
        CounterType ct = counterFromEvent(ev);
        Lane lane = laneOfEvent(ev);
        unsigned inc = (ct == CT_VA_VDST && hasMatrixScalePair(inst)) ? 2u : 1u;
        issued[lane] += inc;
        unsigned ord = issued[lane];
        pendingEvents.insert(ev);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   stamp lane=" << laneName(lane)
                             << " event=" << eventName(ev) << " inc=" << inc << " ord=" << ord
                             << " issued=" << issued[lane]
                             << " (mnemonic=" << inst.getHwInstDesc()->mnemonic << ")\n");

        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        auto stampVA = [&](unsigned idx, HighBitSel half) {
            RegKey k = keyer.producerKey(idx, half);
            VgprStamp& s = scores[k];
            s.vaLane = lane;
            s.vaOrdinal = ord;
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     stamp va v" << k.idx << "("
                                 << halfName(k.half) << ") lane=" << laneName(lane)
                                 << " ord=" << ord << "\n");
        };
        // VM_VSRC ordering-group bookkeeping. A flat_* op is placed in BOTH
        // FIFOs, so it bumps both group counters; pure LDS/VMEM bumps only its
        // own. ordA/ordB are the producer's in-order position within each group
        // it belongs to (0 = not a member).
        unsigned ordA = 0, ordB = 0;
        if (ct == CT_VM_VSRC) {
            if (vmInGroupA(ev)) ordA = ++vmGroupIssued[VM_GROUP_A];
            if (vmInGroupB(ev)) ordB = ++vmGroupIssued[VM_GROUP_B];
        }

        auto stampVM = [&](unsigned idx, HighBitSel half) {
            RegKey k = keyer.producerKey(idx, half);
            VgprStamp& s = scores[k];
            s.vmLane = lane;
            s.vmOrdinal = ord;
            s.vmOrdA = ordA;
            s.vmOrdB = ordB;
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     stamp vm v" << k.idx << "("
                                 << halfName(k.half) << ") lane=" << laneName(lane)
                                 << " ord=" << ord << " ordA=" << ordA << " ordB=" << ordB
                                 << "\n");
        };

        if (ct == CT_VA_VDST) {
            forEachVGPR(
                inst.getSrcRegs(), [&](size_t i) { return srcHalfSel(true16Mod, i); },
                [&](unsigned idx, HighBitSel half) { stampVA(idx, half); });
            forEachVGPR(
                inst.getDestRegs(), [&](size_t i) { return destHalfSel(true16Mod, i); },
                [&](unsigned idx, HighBitSel half) { stampVA(idx, half); });
        } else {
            // VM_VSRC tracks in-flight VMEM reads, which are always full DWORD.
            forEachVGPR(
                inst.getSrcRegs(), [](size_t) { return HighBitSel::NONE; },
                [&](unsigned idx, HighBitSel half) { stampVM(idx, half); });
        }
    }

    // For each VGPR src (RAW on VA_VDST) and each VGPR dst (WAW on VA_VDST,
    // WAR on VM_VSRC), probe the stamp map and accumulate the worst-case wait.
    void onConsumer(const StinkyInstruction& inst, const VGPRHalfKeyer& keyer, Wait& wait) const {
        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        forEachVGPR(
            inst.getSrcRegs(), [&](size_t i) { return srcHalfSel(true16Mod, i); },
            [&](unsigned idx, HighBitSel half) {
                keyer.forEachConsumerKey(
                    idx, half, [&](RegKey k) { determineWait(CT_VA_VDST, k, wait, "src(RAW)"); });
            });

        forEachVGPR(
            inst.getDestRegs(), [&](size_t i) { return destHalfSel(true16Mod, i); },
            [&](unsigned idx, HighBitSel half) {
                keyer.forEachConsumerKey(
                    idx, half, [&](RegKey k) { determineWait(CT_VA_VDST, k, wait, "dst(WAW)"); });
                // WAR on VM_VSRC: writer-vs-in-flight-VMEM-read uses full DWORD.
                RegKey full{RegType::V, idx, RegHalf::NONE};
                determineWait(CT_VM_VSRC, full, wait, "dst(WAR)");
            });
    }

    // Same-group follower count for a VM_VSRC producer stamp. A flat_* producer
    // is in both groups; it benefits from followers in EITHER (whichever FIFO's
    // in-order guarantee proves it done first ⇒ take the max). Non-member groups
    // contribute 0.
    unsigned vmFollowers(const VgprStamp& s) const {
        unsigned fA = s.vmOrdA ? (vmGroupIssued[VM_GROUP_A] - s.vmOrdA) : 0u;
        unsigned fB = s.vmOrdB ? (vmGroupIssued[VM_GROUP_B] - s.vmOrdB) : 0u;
        return std::max(fA, fB);
    }

    // Emit a wait for the hazard on VGPR `k` against counter `c`. The producer's
    // stamp gives the sub-pipe (VA_VDST) or ordering group (VM_VSRC) and the
    // in-order position; the wait value is the same-pipe/same-group follower
    // count. Other pipes/groups only inflate the shared HW counter, so the
    // follower count is a safe upper bound in every case; the only drain is
    // f==0 (producer is the last op of its pipe/group).
    void determineWait(CounterType c, const RegKey& k, Wait& wait, const char* role) const {
        auto it = scores.find(k);
        if (it == scores.end()) return;
        const VgprStamp& s = it->second;

        if (c == CT_VM_VSRC) {
            // Per-ordering-group follower count. Group membership is by
            // instruction type (flat_* ∈ both). If the producer is proven
            // drained in every group it belongs to, no wait.
            bool liveA = s.vmOrdA && s.vmOrdA > vmGroupFloor[VM_GROUP_A];
            bool liveB = s.vmOrdB && s.vmOrdB > vmGroupFloor[VM_GROUP_B];
            if (!liveA && !liveB) return;  // no producer / proven done
            unsigned f = vmFollowers(s);
            unsigned chosen = (f > 0) ? std::min(f, maxEmittableWait(c)) : 0u;
            addWait(wait, c, chosen);
            PASS_DEBUG(std::cerr
                       << "[InsertWaitAlu]     wait hit vm_vsrc on v" << k.idx << "("
                       << halfName(k.half) << "," << role << ") ordA=" << s.vmOrdA
                       << " ordB=" << s.vmOrdB << " issuedA=" << vmGroupIssued[VM_GROUP_A]
                       << " issuedB=" << vmGroupIssued[VM_GROUP_B] << " floorA="
                       << vmGroupFloor[VM_GROUP_A] << " floorB=" << vmGroupFloor[VM_GROUP_B]
                       << " f=" << f << " → wait=" << chosen << "\n");
            return;
        }

        Lane lane = s.vaLane;
        unsigned ord = s.vaOrdinal;
        if (ord == 0 || ord <= floor_[lane]) return;  // no producer / proven done

        unsigned f = issued[lane] - ord;  // same-lane followers, all still in flight
        bool ooo = counterOutOfOrder(c);
        // Per-pipe follower count is safe regardless of how many other VALU
        // sub-pipes are pending: if the producer were still outstanding, all
        // f of its same-pipe followers would be too (FIFO within a pipe), so
        // the counter would exceed f. Other pipes only add to the total, so
        // va_vdst(f) still guarantees the producer is done. This replaces the
        // legacy "drain to 0 when >=2 pipes pending".
        unsigned chosen = std::min(f, maxEmittableWait(c));
        addWait(wait, c, chosen);

        PASS_DEBUG(
            std::cerr << "[InsertWaitAlu]     wait hit " << counterName(c) << " on v" << k.idx
                      << "(" << halfName(k.half) << "," << role << ") lane=" << laneName(lane)
                      << " ord=" << ord << " issued=" << issued[lane] << " floor=" << floor_[lane]
                      << " f=" << f << " ooo=" << ooo
                      << (ooo ? " events={" +
                                    pendingEventsStr(pendingEvents & eventsForCounter(c)) + "}"
                              : std::string())
                      << " → wait=" << chosen << "\n");
    }

    bool counterOutOfOrder(CounterType c) const {
        WaitEventSet ev = pendingEvents & eventsForCounter(c);
        return ev.twoOrMore();
    }

    // Advance floor after a wait is inserted. A wait(count) bounds the TOTAL
    // outstanding across the counter's lanes, so it lifts every lane's floor by
    // (issued - count); count==0 fully drains all the counter's lanes and clears
    // its pending event bits so counterOutOfOrder() no longer flags it.
    void applyWaitcnt(CounterType c, unsigned count) {
        if (count == kNoWait) return;
        for (int L = counterLaneLo(c); L < counterLaneHi(c); ++L) {
            unsigned oldFloor = floor_[L];
            unsigned newFloor = issued[L] >= count ? issued[L] - count : 0u;
            if (newFloor > floor_[L]) floor_[L] = newFloor;
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     apply " << counterName(c) << "(" << count
                                 << ") lane=" << laneName(static_cast<Lane>(L)) << " floor "
                                 << oldFloor << "→" << floor_[L] << " issued=" << issued[L]
                                 << "\n");
        }
        // A vm_vsrc(count) bounds the TOTAL outstanding across both groups, so at
        // most `count` remain in any single group ⇒ each group's floor rises to
        // issued-count. Conservative (never over-raises): count==0 drains both.
        if (c == CT_VM_VSRC) {
            for (int g = 0; g < NUM_VM_GROUPS; ++g) {
                unsigned newFloor =
                    vmGroupIssued[g] >= count ? vmGroupIssued[g] - count : 0u;
                if (newFloor > vmGroupFloor[g]) vmGroupFloor[g] = newFloor;
            }
        }
        if (count == 0) pendingEvents = pendingEvents & ~eventsForCounter(c);
    }

    // Widen this entry state with a predecessor's exit. Returns true (strictDom)
    // when the other side contributed deeper in-flight, a later ordinal, or a
    // new pending event type.
    bool merge(const WaitcntBrackets& other) {
        bool strictDom = false;
        std::array<unsigned, NUM_LANES> myShift{}, otherShift{}, myOldFloor{}, otherOldFloor{};

        for (int L = 0; L < NUM_LANES; ++L) {
            unsigned mineIF = issued[L] - floor_[L];
            unsigned otherIF = other.issued[L] - other.floor_[L];
            unsigned newIssued = floor_[L] + std::max(mineIF, otherIF);
            myOldFloor[L] = floor_[L];
            otherOldFloor[L] = other.floor_[L];
            myShift[L] = newIssued - issued[L];
            otherShift[L] = newIssued - other.issued[L];
            issued[L] = newIssued;
        }

        // Same widening for the VM_VSRC ordering-group counts, so the group
        // ordinals stamped in each predecessor stay comparable after the join.
        std::array<unsigned, NUM_VM_GROUPS> gMyShift{}, gOtherShift{}, gMyOldFloor{},
            gOtherOldFloor{};
        for (int g = 0; g < NUM_VM_GROUPS; ++g) {
            unsigned mineIF = vmGroupIssued[g] - vmGroupFloor[g];
            unsigned otherIF = other.vmGroupIssued[g] - other.vmGroupFloor[g];
            unsigned newIssued = vmGroupFloor[g] + std::max(mineIF, otherIF);
            gMyOldFloor[g] = vmGroupFloor[g];
            gOtherOldFloor[g] = other.vmGroupFloor[g];
            gMyShift[g] = newIssued - vmGroupIssued[g];
            gOtherShift[g] = newIssued - other.vmGroupIssued[g];
            vmGroupIssued[g] = newIssued;
        }

        for (const auto& [k, _] : other.scores) scores.try_emplace(k);

        for (auto& [k, s] : scores) {
            auto it = other.scores.find(k);
            const VgprStamp* o = (it != other.scores.end()) ? &it->second : nullptr;
            mergeSide(s.vaLane, s.vaOrdinal, o ? o->vaLane : LANE_CSMACC, o ? o->vaOrdinal : 0,
                      myShift, otherShift, myOldFloor, otherOldFloor, strictDom);
            mergeSide(s.vmLane, s.vmOrdinal, o ? o->vmLane : LANE_LDS, o ? o->vmOrdinal : 0,
                      myShift, otherShift, myOldFloor, otherOldFloor, strictDom);
            mergeGroupOrd(s.vmOrdA, o ? o->vmOrdA : 0, VM_GROUP_A, gMyShift, gOtherShift,
                          gMyOldFloor, gOtherOldFloor, strictDom);
            mergeGroupOrd(s.vmOrdB, o ? o->vmOrdB : 0, VM_GROUP_B, gMyShift, gOtherShift,
                          gMyOldFloor, gOtherOldFloor, strictDom);
        }

        if (!pendingEvents.containsAll(other.pendingEvents)) strictDom = true;
        pendingEvents = pendingEvents | other.pendingEvents;
        return strictDom;
    }

   private:
    // Sum issued/floor across the lanes owned by counter `c`.
    static unsigned laneSum(CounterType c, const std::array<unsigned, NUM_LANES>& a) {
        unsigned n = 0;
        for (int L = counterLaneLo(c); L < counterLaneHi(c); ++L) n += a[L];
        return n;
    }

    // Merge one side (VA or VM) of a VGPR stamp: shift both producers past their
    // floors and keep the later (more conservative) one.
    static void mergeSide(Lane& myLane, unsigned& myOrd, Lane oLane, unsigned oOrd,
                          const std::array<unsigned, NUM_LANES>& myShift,
                          const std::array<unsigned, NUM_LANES>& otherShift,
                          const std::array<unsigned, NUM_LANES>& myOldFloor,
                          const std::array<unsigned, NUM_LANES>& otherOldFloor, bool& strictDom) {
        unsigned myS = myOrd <= myOldFloor[myLane] ? 0 : myOrd + myShift[myLane];
        unsigned oS = (oOrd && oOrd > otherOldFloor[oLane]) ? oOrd + otherShift[oLane] : 0;
        if (oS > myS) {
            myLane = oLane;
            myOrd = oS;
            strictDom = true;
        } else {
            myOrd = myS;
        }
    }

    // Merge one VM_VSRC ordering-group ordinal of a VGPR stamp. Fixed group `g`
    // (unlike VA lanes, group membership is by instruction type, not tracked in
    // the stamp), so only the ordinal shifts. Keep the later (more conservative)
    // producer position.
    static void mergeGroupOrd(unsigned& myOrd, unsigned oOrd, VMGroup g,
                              const std::array<unsigned, NUM_VM_GROUPS>& myShift,
                              const std::array<unsigned, NUM_VM_GROUPS>& otherShift,
                              const std::array<unsigned, NUM_VM_GROUPS>& myOldFloor,
                              const std::array<unsigned, NUM_VM_GROUPS>& otherOldFloor,
                              bool& strictDom) {
        unsigned myS = (myOrd && myOrd > myOldFloor[g]) ? myOrd + myShift[g] : 0;
        unsigned oS = (oOrd && oOrd > otherOldFloor[g]) ? oOrd + otherShift[g] : 0;
        if (oS > myS) {
            myOrd = oS;
            strictDom = true;
        } else {
            myOrd = myS;
        }
    }

    std::array<unsigned, NUM_LANES> issued = {};
    std::array<unsigned, NUM_LANES> floor_ = {};
    // VM_VSRC per-ordering-group issued/floor. Parallel to issued/floor_ but
    // indexed by VMGroup, because flat_* bumps both groups at once (which the
    // per-lane arrays cannot express). Consulted only on the vm_vsrc decision.
    std::array<unsigned, NUM_VM_GROUPS> vmGroupIssued = {};
    std::array<unsigned, NUM_VM_GROUPS> vmGroupFloor = {};
    WaitEventSet pendingEvents;
    std::unordered_map<RegKey, VgprStamp, RegKeyHash> scores;
};

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

class InsertWaitAluPassImpl : public Pass {
    StinkyAsmModule* module = nullptr;
    std::unordered_map<BasicBlock*, WaitcntBrackets> blockEntryState;
    GfxArchID archId = GfxArchID{};
    VGPRHalfKeyer keyer{};

   public:
    explicit InsertWaitAluPassImpl(StinkyAsmModule* module) : module(module) {}

   private:
    StinkyInstruction* emitWaitAlu(BasicBlock& bb, IRBase* insertBefore, const Wait& wait,
                                   int hold_cnt = -1) {
        AsmIRBuilder builder(bb, archId);
        StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_alu, archId), insertBefore);
        int va = isNoWait(wait, CT_VA_VDST) ? -1 : static_cast<int>(wait.get(CT_VA_VDST));
        int vm = isNoWait(wait, CT_VM_VSRC) ? -1 : static_cast<int>(wait.get(CT_VM_VSRC));
        w->addModifier<SWaitAluData>(SWaitAluData(va, /*va_sdst=*/-1, /*va_ssrc=*/-1, hold_cnt, vm,
                                                  /*va_vcc=*/-1,
                                                  /*sa_sdst=*/-1));
        return w;
    }

    // If the instruction immediately before `consumer` in `bb` is a hold_cnt-only
    // s_wait_alu survivor, return its hold_cnt value and erase the instruction
    // so the caller can fold the hold_cnt into a freshly-emitted merged wait.
    // Returns -1 if no such survivor is adjacent.
    //
    // Scope note: this only handles the hold_cnt-only shape because that is
    // the only pre-existing s_wait_alu RemoveWaitAluPass leaves behind. A
    // general per-field min merge across non-trivial va_vdst/vm_vsrc would
    // need more care and isn't needed today.
    int extractAdjacentHoldCnt(BasicBlock& bb, IRBase* consumer) {
        auto consumerIt = IRList::iterator(consumer);
        if (consumerIt == bb.begin()) return -1;
        auto prevIt = consumerIt;
        --prevIt;
        auto* prev = dyn_cast<StinkyInstruction>(prevIt.getNodePtr());
        if (!prev || !isWaitAluInst(*prev)) return -1;
        auto* data = prev->getModifier<SWaitAluData>();
        if (!data) return -1;
        if (!data->hasField(SWaitAluData::HOLD_CNT)) return -1;
        if (data->hasField(SWaitAluData::VA_VDST)) return -1;
        if (data->hasField(SWaitAluData::VM_VSRC)) return -1;
        int hold_cnt = static_cast<int>(data->getField(SWaitAluData::HOLD_CNT));
        bb.eraseIR(prevIt);
        return hold_cnt;
    }

    Wait computeWaitForInst(const StinkyInstruction& inst, const WaitcntBrackets& sb) const {
        Wait wait;

        // Step 1: scoreboard probes on every VGPR operand of `inst`.
        sb.onConsumer(inst, keyer, wait);

        // Step 2: skip VA_VDST for VALU consumers
        if (isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst)) {
            if (!isNoWait(wait, CT_VA_VDST))
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     suppress va_vdst (VALU consumer, was "
                                     << int(wait.get(CT_VA_VDST)) << ")\n");
            setNoWait(wait, CT_VA_VDST);
        }

        // Step 3: eager EXEC guard. If this instruction modifies EXEC and any
        // VA_VDST work is in flight, drain now — subsequent VALUs may be
        // EXEC-skipped at runtime and therefore won't bump VA_VDST_hw, leaving
        // any precomputed non-zero wait invalid. Must run AFTER Step 2 so that
        // v_cmpx_* (VALU + writes EXEC) gets the va_vdst(0) drain rather than
        // the VALU suppression.
        if (writesExec(inst) && sb.getScoreRange(CT_VA_VDST) > 0) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     drain va_vdst (EXEC writer, in-flight="
                                 << sb.getScoreRange(CT_VA_VDST) << ")\n");
            addWait(wait, CT_VA_VDST, 0);
        }

        return wait;
    }

    // Process one BB starting from its accumulated entry state.
    // emit=false → run scoreboard, return exit state for Phase 1 propagation.
    // emit=true → re-run with the converged entry state and insert s_wait_alu.
    WaitcntBrackets runOnBasicBlock(BasicBlock& bb, bool emit) {
        WaitcntBrackets sb = blockEntryState[&bb];

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] " << (emit ? "emit" : "analyze") << " bb=\""
                             << bb.getLabel()
                             << "\" entry=[va_vdst LB=" << sb.getScoreLB(CT_VA_VDST)
                             << " UB=" << sb.getScoreUB(CT_VA_VDST) << " sz=" << sb.scoresSize()
                             << "; vm_vsrc LB=" << sb.getScoreLB(CT_VM_VSRC)
                             << " UB=" << sb.getScoreUB(CT_VM_VSRC) << "]\n");

        for (auto it = bb.begin(); it != bb.end();) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst) {
                ++it;
                continue;
            }
            if (isPseudoInst(inst)) {
                ++it;
                continue;
            }

            // Pre-existing s_wait_alu: absorb its va_vdst/vm_vsrc into LB so the
            // rest of the BB sees the post-wait state, and leave the instruction
            // in place so the runtime drain actually happens. Today the only
            // realistic source is hold_cnt-only survivors from RemoveWaitAluPass
            // (their va_vdst/vm_vsrc are already kNoWait, so the absorb is a
            // no-op); the emit branch below merges fresh va_vdst/vm_vsrc into
            // such a survivor when it's the immediately-preceding instruction.
            if (isWaitAluInst(*inst)) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]   absorb existing s_wait_alu\n");
                if (const auto* data = inst->getModifier<SWaitAluData>()) {
                    if (data->hasField(SWaitAluData::VA_VDST))
                        sb.applyWaitcnt(CT_VA_VDST, data->getField(SWaitAluData::VA_VDST));
                    if (data->hasField(SWaitAluData::VM_VSRC))
                        sb.applyWaitcnt(CT_VM_VSRC, data->getField(SWaitAluData::VM_VSRC));
                }
                ++it;
                continue;
            }

            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   visit " << inst->getHwInstDesc()->mnemonic
                                 << "\n");

            // Function call (s_swappc): drain both counters right after the call,
            // at the return-landing site. The callee may leave VALU/VMEM
            // instructions outstanding on VA_VDST/VM_VSRC, so the drain is
            // unconditional. The callee entry is drained separately
            // (runCalleeConservativeDrain).
            if (isCall(*inst)) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]   call — drain va_vdst(0)+vm_vsrc(0) "
                                        "after s_swappc (callee->caller bracket)\n");
                // nextIt is the instruction after the call. The drain is inserted
                // before it, so resuming at nextIt continues past the drain
                // instead of re-visiting it.
                auto nextIt = it;
                ++nextIt;
                if (emit) {
                    Wait drain;
                    addWait(drain, CT_VA_VDST, 0);
                    addWait(drain, CT_VM_VSRC, 0);
                    // Insert before the node after the call (append at BB end if
                    // the call is the last node) so the drain lands in the
                    // caller's own BB, bound to the return path.
                    IRBase* insertBefore = (nextIt == bb.end()) ? nullptr : nextIt.getNodePtr();
                    emitWaitAlu(bb, insertBefore, drain);
                }
                sb.applyWaitcnt(CT_VA_VDST, 0);
                sb.applyWaitcnt(CT_VM_VSRC, 0);
                it = nextIt;
                continue;
            }

            Wait wait = computeWaitForInst(*inst, sb);
            if (wait.hasAny()) {
                PASS_DEBUG(std::cerr
                           << "[InsertWaitAlu]   emit s_wait_alu va_vdst="
                           << (isNoWait(wait, CT_VA_VDST) ? -1 : int(wait.get(CT_VA_VDST)))
                           << " vm_vsrc="
                           << (isNoWait(wait, CT_VM_VSRC) ? -1 : int(wait.get(CT_VM_VSRC)))
                           << "\n");
                if (emit) {
                    // If the immediately-preceding instruction is a hold_cnt-only
                    // s_wait_alu survivor, fold its hold_cnt into our new wait
                    // so the constraint isn't lost and we don't emit two
                    // adjacent waits.
                    int holdCnt = extractAdjacentHoldCnt(bb, inst);
                    if (holdCnt >= 0)
                        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     fold hold_cnt=" << holdCnt
                                             << " from adjacent survivor\n");
                    emitWaitAlu(bb, inst, wait, holdCnt);
                    PASS_DEBUG(std::cerr << "[InsertWaitAlu]     inserted s_wait_alu before "
                                         << inst->getHwInstDesc()->mnemonic << "\n");
                }
                if (!isNoWait(wait, CT_VA_VDST)) sb.applyWaitcnt(CT_VA_VDST, wait.get(CT_VA_VDST));
                if (!isNoWait(wait, CT_VM_VSRC)) sb.applyWaitcnt(CT_VM_VSRC, wait.get(CT_VM_VSRC));
            }

            if (auto ev = classifyEvent(*inst)) sb.onProducer(*ev, *inst, keyer);

            ++it;
        }

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] end-of-bb \"" << bb.getLabel()
                             << "\" sb=[va_vdst LB=" << sb.getScoreLB(CT_VA_VDST)
                             << " UB=" << sb.getScoreUB(CT_VA_VDST) << " sz=" << sb.scoresSize()
                             << "; vm_vsrc LB=" << sb.getScoreLB(CT_VM_VSRC)
                             << " UB=" << sb.getScoreUB(CT_VM_VSRC) << "]\n");
        return sb;
    }

    // Build "s_setreg_imm32_b32 hwreg(SCHED_MODE, DEP_MODE), value"
    StinkyInstruction* makeSchedModeSetreg(BasicBlock& bb, IRBase* insertBefore, int value) {
        AsmIRBuilder builder(bb, archId);
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::s_setreg_IMM32_b32, archId), insertBefore);
        const HwReg::SubField depMode = HwReg::schedModeDepMode(archId);
        inst->addDestReg(
            StinkyRegister::Hwreg(HwReg::schedModeId(archId), depMode.offset, depMode.size));
        inst->addSrcReg(StinkyRegister(value));
        return inst;
    }

    void insertSchedModeLifecycle(Function& func) {
        BasicBlock* entry = func.getEntryBlock();
        if (!entry) return;

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 3: insert mode2 enable setreg\n");

        // Whole-kernel mode2: enable at the kernel entry label(s). Mode2 stays
        // active across function calls and across the whole kernel body — it is
        // never switched back to mode0.

        // The wave can enter the compute region through two labels: the
        // kernarg-preload path jumps straight to label_Preload_Offset_Start
        // (skipping the +0..255 prologue), while the non-preload path enters at
        // label_ASM_Start (the main-body entry). A kernel may emit either or
        // both. Enable mode2 at EVERY entry label present so whichever path the
        // wave takes hits a setreg(SCHED_MODE)=2 — re-enabling is idempotent and
        // the span between the two labels is SALU kernarg processing (no
        // VALU/VMEM in flight), so a second enable is still drain-free.
        // If no entry label is found, fall back to the function entry block.
        std::vector<BasicBlock*> anchorBBs;
        for (BasicBlock& bb : func) {
            if (bb.getLabel() == "label_Preload_Offset_Start" ||
                bb.getLabel() == "label_ASM_Start") {
                anchorBBs.push_back(&bb);
            }
        }
        if (anchorBBs.empty()) anchorBBs.push_back(entry);

        // Drain-free: each anchor is a kernel entry (all DEPCTR counters zero,
        // SALU kernarg code follows).
        for (BasicBlock* anchorBB : anchorBBs) {
            // Skip leading labels / pseudo instructions so the setreg lands at
            // the first real instruction position after the label.
            auto anchorIt = anchorBB->begin();
            while (anchorIt != anchorBB->end()) {
                auto* inst = dyn_cast<StinkyInstruction>(anchorIt.getNodePtr());
                if (inst && isPseudoInst(inst)) {
                    ++anchorIt;
                    continue;
                }
                break;
            }
            IRBase* anchor = (anchorIt == anchorBB->end()) ? nullptr : anchorIt.getNodePtr();
            makeSchedModeSetreg(*anchorBB, anchor, /*value=*/2);
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   inserted setreg(SCHED_MODE)=2 at entry "
                                    "bb=\""
                                 << anchorBB->getLabel() << "\"\n");
        }
    }

   public:
    static char ID;
    const char* getName() const override {
        return "InsertWaitAluPass";
    }
    Pass::ID getPassID() const override {
        return &InsertWaitAluPassImpl::ID;
    }

   private:
    // Conservatively drain both counters at the callee entry.
    //
    // TODO: also run the full fixed-point WaitAlu analysis over the callee body
    // (build its CFG, run the scoreboard) so the callee gets its own intra-body
    // s_wait_alu, matching the entry-function path. Today the callee body is left
    // un-analyzed.
    void runCalleeConservativeDrain(Function& callee) {
        BasicBlock* entry = callee.getEntryBlock();
        if (!entry) return;

        // A callee that never touches a VGPR (e.g. the "None" activation, which
        // only does s_setpc back) has no hazard to drain — skip it.
        if (!functionReadsOrWritesVGPR(callee)) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu] callee \"" << callee.getName()
                                 << "\": no VGPR use, skip entry drain\n");
            return;
        }

        // Land the drain at the first real instruction.
        auto it = entry->begin();
        while (it != entry->end()) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst && isPseudoInst(inst)) {
                ++it;
                continue;
            }
            break;
        }
        IRBase* anchor = (it == entry->end()) ? nullptr : it.getNodePtr();

        Wait drain;
        addWait(drain, CT_VA_VDST, 0);
        addWait(drain, CT_VM_VSRC, 0);
        emitWaitAlu(*entry, anchor, drain);
        PASS_DEBUG(std::cerr << "[InsertWaitAlu] callee \"" << callee.getName()
                             << "\": entry drain va_vdst(0)+vm_vsrc(0)\n");
    }

    // True if any real instruction in func reads or writes a VGPR.
    static bool functionReadsOrWritesVGPR(Function& func) {
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;
                for (const auto& r : inst->getSrcRegs())
                    if (r.dataType == StinkyRegister::Type::Register && r.reg.type == RegType::V)
                        return true;
                for (const auto& r : inst->getDestRegs())
                    if (r.dataType == StinkyRegister::Type::Register && r.reg.type == RegType::V)
                        return true;
            }
        }
        return false;
    }

    // Full fixed-point scoreboard analysis + mode2 enable for the entry
    // (non-callee) function.
    void runEntryFunction(Function& func, AnalysisManager& AM) {
        const auto& bbIndex = AM.getResult<BBIndexAnalysis>(func);
        const auto& rpo = bbIndex.rpo;

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 1: fixed-point analysis (" << rpo.size()
                             << " BBs in RPO)\n");

        // Phase 1: fixed-point analysis using entry-state propagation.
        // Each BB starts from its accumulated entry state. After processing,
        // the exit state is merged into each successor's entry state. The
        // merge is monotonically widening, guaranteeing convergence.
        {
            std::vector<BasicBlock*> worklist;
            std::unordered_set<BasicBlock*> inWL;
            for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
                worklist.push_back(*it);
                inWL.insert(*it);
            }
            unsigned visits = 0;
            while (!worklist.empty()) {
                BasicBlock* bb = worklist.back();
                worklist.pop_back();
                inWL.erase(bb);
                ++visits;
                WaitcntBrackets exitState = runOnBasicBlock(*bb, /*emit=*/false);
                for (auto* succ : bb->getSuccessors()) {
                    if (blockEntryState[succ].merge(exitState)) {
                        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   entry widened for bb=\""
                                             << succ->getLabel() << "\" — queueing\n");
                        if (inWL.insert(succ).second) worklist.push_back(succ);
                    }
                }
            }
            PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 1 converged after " << visits
                                 << " BB visits\n");
        }

        // Phase 2: emit s_wait_alu using converged state. Caller->callee bracket
        // drains are emitted here, in the isCall branch of runOnBasicBlock.
        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 2: emit s_wait_alu instructions\n");
        for (auto* bb : rpo) runOnBasicBlock(*bb, /*emit=*/true);

        // Phase 3: enable mode2 at entry label (never disabled thereafter).
        insertSchedModeLifecycle(func);

        blockEntryState.clear();
    }

   public:
    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const auto* archInfo = ArchHelper::getInstance().getArchInfo(archId);
        const bool hasD16 = archInfo && archInfo->hasD16Writes32BitVgpr();
        keyer = VGPRHalfKeyer(hasD16);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << " hasD16Writes32BitVgpr=" << hasD16 << "\n");

        // Whole-kernel: process the entry function with full analysis, then apply
        // the conservative entry drain to every callee. The pass is invoked on the
        // entry function; callees are reached via the module. Guard against being
        // re-invoked per-function by a future driver: only the non-callee run
        // drives callee processing.
        if (func.getIsCallable()) {
            if (!func.empty()) runCalleeConservativeDrain(func);
            return PreservedAnalyses::none();
        }

        if (!func.empty()) runEntryFunction(func, AM);

        // Reach callees only when a module is available (backend pipeline). In
        // stinkytofu-opt single-pass mode / unit tests there is no module.
        if (module) {
            for (Function* fn : module->getFunctions()) {
                if (fn && fn->getIsCallable() && !fn->empty()) runCalleeConservativeDrain(*fn);
            }
        }

        return PreservedAnalyses::none();
    }
};

char InsertWaitAluPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertWaitAluPass(StinkyAsmModule& module) {
    return std::make_unique<InsertWaitAluPassImpl>(&module);
}
std::unique_ptr<Pass> createInsertWaitAluPass() {
    return std::make_unique<InsertWaitAluPassImpl>(nullptr);
}
}  // namespace stinkytofu
