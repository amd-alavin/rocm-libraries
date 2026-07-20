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

#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "InsertCoexecHazardPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

// Per-arch co-execution hazard configuration.
//
// The WMMA V_NOP counts are NOT tabulated here: they are derived at runtime from
// each producer's HwInstDesc coIssueWindow bitmask (popcount == coexec slots).
// Only the arch-level rules that cannot be read off a single instruction live in
// this struct.
struct CoexecHazardConfig {
    // TRANS -> TRANS and TRANS -> XDL WMMA spacing (independent op / V_NOP).
    int transToNonCoreSide = 0;
    // Whether HW detects TRANS -> core/side VALU (then no V_NOP is needed).
    bool hwHandlesTransToCoreSide = false;
    // Whether the arch has the SCHED_MODE DISABLE_XDL_ARB_STALL bit. When set at
    // runtime co-execution is OFF and the reduced counts apply.
    bool hasArbStallBit = false;
};

// MI450-B0 / gfx1250.
constexpr CoexecHazardConfig kGfx1250Config = {
    /*transToNonCoreSide=*/1,
    /*hwHandlesTransToCoreSide=*/true,
    /*hasArbStallBit=*/true,
};

// Largest slot budget worth scanning: max WMMA->WMMA count on gfx1250 is
// popcount(0xFF00)+1 = 9. 18 leaves generous headroom and bounds the scan.
constexpr int kMaxSlotBudget = 18;

enum class ProducerKind { WMMA, TRANS };

// What the current consumer is looking for during a backward scan.
struct ConsumerCtx {
    ProducerKind kind;
    bool consumerIsWmma;  // only meaningful for kind == WMMA
    const StinkyInstruction* consumer;
    bool coexecOff;
};

inline int popcount16(uint16_t v) {
    return __builtin_popcount(static_cast<unsigned>(v));
}

// Does this instruction fill a VALU co-execution [I] slot? Only VALU-pipe ops
// (incl. transcendental, matrix, and bare v_nop) count; S_NOP/S_WAIT_ALU/
// S_DELAY_ALU/SALU/VMEM do not.
inline bool isSlotFiller(const StinkyInstruction& inst) {
    return isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst) ||
           inst.getUnifiedOpcode() == GFX::v_nop;
}

// A co-executable VALU consumer: a VALU (incl. transcendental) that is not a
// matrix instruction. Matches LLVM's isCoexecutableVALUInst intent.
inline bool isCoexecutableVALU(const StinkyInstruction& inst) {
    return (isVectorALU(inst) || isTranscendental(inst)) && !isMatrixInstruction(inst);
}

// WMMA producer D feeds a WMMA consumer's A/B (or SWMMAC index). D->C
// (accumulation) is intentionally NOT a hazard — HW forwards it.
bool wmmaToWmmaOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    const auto& srcs = cons.getSrcRegs();
    if (srcs.size() > 0 && d.isOverlap(srcs[0])) return true;  // A
    if (srcs.size() > 1 && d.isOverlap(srcs[1])) return true;  // B
    if (isSWMMA(cons) && srcs.size() > 2 && d.isOverlap(srcs[2])) return true;  // index
    return false;
}

// WMMA producer D vs a co-executable VALU consumer: RAW (D->src), WAW (D->dst),
// WAR (producer A/B -> consumer dst).
bool wmmaToValuOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    for (const StinkyRegister& s : cons.getSrcRegs())
        if (d.isOverlap(s)) return true;  // RAW
    for (const StinkyRegister& cd : cons.getDestRegs())
        if (d.isOverlap(cd)) return true;  // WAW
    const auto& psrc = prod.getSrcRegs();
    for (size_t i = 0; i < psrc.size() && i < 2; ++i)  // A, B
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (psrc[i].isOverlap(cd)) return true;  // WAR
    return false;
}

// TRANS producer vs consumer: RAW/WAW on producer dst, WAR on producer src.
bool transOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    for (const StinkyRegister& d : prod.getDestRegs()) {
        for (const StinkyRegister& s : cons.getSrcRegs())
            if (d.isOverlap(s)) return true;  // RAW
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (d.isOverlap(cd)) return true;  // WAW
    }
    for (const StinkyRegister& ps : prod.getSrcRegs())
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (ps.isOverlap(cd)) return true;  // WAR
    return false;
}

class InsertCoexecHazardPass : public StinkyInstPass {
   public:
    static char ID;
    explicit InsertCoexecHazardPass(StinkyAsmModule* module) : module_(module) {}

    const char* getName() const override {
        return "InsertCoexecHazardPass";
    }

    PassID getPassID() const override {
        return &InsertCoexecHazardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId_ = getGfxArchID(arch[0], arch[1], arch[2]);
        config_ = kGfx1250Config;

        PASS_DEBUG(std::cerr << "[InsertCoexecHazard] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << "\n");

        // Whole-kernel: process the entry function, then every callee. The pass
        // is invoked on the entry function; callees are reached via the module.
        // If a future driver invokes it directly on a callable, just process it.
        if (func.getIsCallable()) {
            if (!func.empty()) processFunction(func);
            return preserveCFGAnalyses();
        }

        if (!func.empty()) processFunction(func);

        if (module_) {
            for (Function* fn : module_->getFunctions())
                if (fn && fn->getIsCallable() && !fn->empty()) processFunction(*fn);
        }

        return preserveCFGAnalyses();
    }

   private:
    // Detect TensileLite's `s_setreg hwreg(SCHED_MODE, offset=4, size=1), 1`
    // which disables the XDL arb stall (co-execution OFF from here on).
    bool isArbStallSetreg(const StinkyInstruction& inst) const {
        if (!config_.hasArbStallBit) return false;
        return HwReg::isSetregTo(inst, HwReg::schedModeId(archId_),
                                 HwReg::schedModeDisableXdlArbStall(archId_));
    }

    // V_NOPs a consumer needs behind a matched producer.
    int required(ProducerKind kind, int slots, bool consumerIsWmma, bool off) const {
        if (kind == ProducerKind::TRANS) return config_.transToNonCoreSide;
        // WMMA producer.
        if (off) return consumerIsWmma ? 1 : 0;
        return consumerIsWmma ? slots + 1 : slots;
    }

    // Does `prod` match what `ctx` is scanning for?
    bool matches(const StinkyInstruction& prod, const ConsumerCtx& ctx) const {
        if (ctx.kind == ProducerKind::WMMA) {
            if (!isXDLWMMA(prod)) return false;
            return ctx.consumerIsWmma ? wmmaToWmmaOverlap(prod, *ctx.consumer)
                                      : wmmaToValuOverlap(prod, *ctx.consumer);
        }
        if (!isTranscendental(prod)) return false;
        return transOverlap(prod, *ctx.consumer);
    }

    // Real instructions of `bb` in program order (skips pseudo-insts). Scans use
    // indices into this vector so they never rely on decrementing end(), which in
    // this intrusive list is a null sentinel whose operator-- is a no-op.
    static std::vector<StinkyInstruction*> realInsts(BasicBlock& bb) {
        std::vector<StinkyInstruction*> out;
        for (auto& node : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst && !isPseudoInst(inst)) out.push_back(inst);
        }
        return out;
    }

    // Backward scan of `bb` starting strictly before `startBefore` (or from the
    // end of the block when `startBefore` is null, used when entering a
    // predecessor). Follows predecessors while the slot budget is not exhausted.
    // Returns the maximum (required - existing) over every matching producer
    // reachable within budget; a non-positive result means no insertion is
    // needed. `accExisting` is the slot-filler count already accumulated between
    // the consumer and the scan start. `visited` is persistent (never erased) so
    // each BB is scanned at most once per consumer query — bounding the traversal
    // to O(#BB) and preventing exponential path re-entry on a real kernel CFG.
    //
    // The scan start is located by pointer identity in a freshly collected inst
    // vector, so v_nops inserted before earlier consumers in this same block are
    // correctly counted as existing slot-fillers for later consumers.
    int scanBack(BasicBlock& bb, const StinkyInstruction* startBefore, int accExisting,
                 const ConsumerCtx& ctx, std::unordered_set<const BasicBlock*>& visited) {
        int best = INT_MIN;
        int existing = accExisting;

        const std::vector<StinkyInstruction*> insts = realInsts(bb);
        int start = static_cast<int>(insts.size());
        if (startBefore) {
            for (int i = 0; i < static_cast<int>(insts.size()); ++i)
                if (insts[i] == startBefore) {
                    start = i;
                    break;
                }
        }

        for (int i = start - 1; i >= 0; --i) {
            StinkyInstruction& inst = *insts[i];

            // A call is a hard boundary: do not scan across it (cross-call drains
            // are handled conservatively elsewhere).
            if (isCall(inst)) return best;

            if (matches(inst, ctx)) {
                const int slots =
                    ctx.kind == ProducerKind::WMMA ? popcount16(inst.getHwInstDesc()->coIssueWindow)
                                                   : 0;
                const int need = required(ctx.kind, slots, ctx.consumerIsWmma, ctx.coexecOff);
                best = std::max(best, need - existing);
            }

            if (isSlotFiller(inst)) ++existing;
            if (existing > kMaxSlotBudget) return best;
        }

        // Reached the top of the BB with budget to spare: continue into
        // predecessors. Each BB is entered at most once (persistent visited set).
        for (BasicBlock* pred : bb.getPredecessors()) {
            if (!visited.insert(pred).second) continue;
            best = std::max(best, scanBack(*pred, /*startBefore=*/nullptr, existing, ctx, visited));
        }
        return best;
    }

    // Compute, for every BB, whether co-execution is OFF on entry: OFF iff every
    // predecessor exits OFF (an arb-stall setreg has executed on all paths). The
    // setreg is monotonic (never re-enabled), so the AND fixed-point converges.
    // A BB treated as ON when it is really OFF only over-inserts V_NOPs (safe);
    // the reverse would under-insert, so ON is the conservative default.
    void computeCoexecOff(Function& func, std::unordered_map<const BasicBlock*, bool>& entryOff) {
        std::unordered_map<const BasicBlock*, bool> hasSetreg, exitOff;
        for (BasicBlock& bb : func) {
            bool found = false;
            for (auto& node : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (inst && !isPseudoInst(inst) && isArbStallSetreg(*inst)) {
                    found = true;
                    break;
                }
            }
            hasSetreg[&bb] = found;
            entryOff[&bb] = false;
            exitOff[&bb] = found;
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (BasicBlock& bb : func) {
                const auto& preds = bb.getPredecessors();
                bool en = !preds.empty();
                for (BasicBlock* p : preds) en = en && exitOff[p];
                if (en != entryOff[&bb]) {
                    entryOff[&bb] = en;
                    changed = true;
                }
                const bool ex = en || hasSetreg[&bb];
                if (ex != exitOff[&bb]) {
                    exitOff[&bb] = ex;
                    changed = true;
                }
            }
        }
    }

    void processFunction(Function& func) {
        std::unordered_map<const BasicBlock*, bool> entryOff;
        computeCoexecOff(func, entryOff);

        for (BasicBlock& bb : func) {
            bool off = entryOff[&bb];
            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) {
                    ++it;
                    continue;
                }

                if (isArbStallSetreg(*inst)) {
                    off = true;
                    ++it;
                    continue;
                }

                int toInsert = 0;
                if (isXDLWMMA(*inst)) {
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::WMMA,
                                                           /*consumerIsWmma=*/true, off));
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::TRANS,
                                                           /*consumerIsWmma=*/false, off));
                } else if (isCoexecutableVALU(*inst)) {
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::WMMA,
                                                           /*consumerIsWmma=*/false, off));
                    // TRANS -> core/side is HW-handled; only a TRANS consumer needs
                    // the TRANS -> TRANS spacing.
                    if (isTranscendental(*inst))
                        toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::TRANS,
                                                               /*consumerIsWmma=*/false, off));
                }

                if (toInsert > 0) {
                    insertVNops(bb, it.getNodePtr(), toInsert);
                    PASS_DEBUG(std::cerr << "[InsertCoexecHazard]   inserted " << toInsert
                                         << " v_nop before " << inst->getHwInstDesc()->mnemonic
                                         << " (coexec " << (off ? "OFF" : "ON") << ")\n");
                }
                ++it;
            }
        }
    }

    int hazardFor(BasicBlock& bb, const StinkyInstruction& consumer, ProducerKind kind,
                  bool consumerIsWmma, bool off) {
        ConsumerCtx ctx{kind, consumerIsWmma, &consumer, off};
        // Do NOT pre-mark the consumer's block visited: on a self-loop it is its
        // own predecessor, and the backedge must be followed once (re-scanning
        // the block from its end) to reach a producer at the loop tail. The
        // persistent visited set still bounds this — the block is scanned at most
        // twice (once from the consumer, once from the end via the backedge).
        std::unordered_set<const BasicBlock*> visited;
        const int r = scanBack(bb, /*startBefore=*/&consumer, /*accExisting=*/0, ctx, visited);
        return r > 0 ? r : 0;
    }

    void insertVNops(BasicBlock& bb, IRBase* insertBefore, int n) {
        AsmIRBuilder builder(bb, archId_);
        for (int i = 0; i < n; ++i) builder.create(getMCIDByUOp(GFX::v_nop, archId_), insertBefore);
    }

    StinkyAsmModule* module_ = nullptr;
    GfxArchID archId_ = GfxArchID{};
    CoexecHazardConfig config_;
};

char InsertCoexecHazardPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertCoexecHazardPass(StinkyAsmModule& module) {
    return std::make_unique<InsertCoexecHazardPass>(&module);
}
std::unique_ptr<Pass> createInsertCoexecHazardPass() {
    return std::make_unique<InsertCoexecHazardPass>(nullptr);
}
}  // namespace stinkytofu
