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

#include <iostream>

#define DEBUG_TYPE "InsertCoexecHazardPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
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
                             << " (scaffold: no hazards inserted yet)\n");

        // Algorithm implemented in a follow-up commit.
        (void)func;
        (void)module_;

        return preserveCFGAnalyses();
    }

   private:
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
