// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fmt/ranges.h>
#include "common/common_types.h"
#include "common/logging.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/memory_manager.h"

namespace VideoCommon {

/// The texture handle a shader reads from a storage buffer, computed from the current constant
/// buffer and storage buffer words (Shader::Info::handle_programs). cbuf_address(index) is the
/// bound constant buffer's address, 0 when none is bound.
template <typename CbufAddress>
u32 EvaluateTextureHandle(const Shader::Info& info, u32 program, Tegra::MemoryManager& gpu_memory,
                          CbufAddress&& cbuf_address) {
    // EDEN_NX2_RENDER_TRACE: report handle programs that evaluate to 0 and why (first 128).
    static const bool trace = std::getenv("EDEN_NX2_RENDER_TRACE") != nullptr;
    const char* missing = nullptr;
    u32 missing_index = 0, missing_offset = 0;
    const auto note = [&](const char* why, u32 index, u32 offset) {
        if (!missing) {
            missing = why;
            missing_index = index;
            missing_offset = offset;
        }
    };
    const u32 handle = Shader::EvaluateHandleProgram(
               info.handle_programs[program],
               [&](u32 index, u32 offset) -> u32 {
                   const GPUVAddr base{cbuf_address(index)};
                   if (base == 0) note("cbuf not bound", index, offset);
                   return base != 0 ? gpu_memory.Read<u32>(base + offset) : 0U;
               },
               [&](u32 binding, u32 offset) -> u32 {
                   if (binding >= info.storage_buffers_descriptors.size()) {
                       note("storage binding out of range", binding, offset);
                       return 0U;
                   }
                   const auto& descriptor{info.storage_buffers_descriptors[binding]};
                   const GPUVAddr cbuf{cbuf_address(descriptor.cbuf_index)};
                   if (cbuf == 0) {
                       note("storage cbuf not bound", binding, offset);
                       return 0U;
                   }
                   const GPUVAddr base{gpu_memory.Read<u64>(cbuf + descriptor.cbuf_offset)};
                   if (base == 0) note("storage address 0", binding, offset);
                   return base != 0 ? gpu_memory.Read<u32>(base + offset) : 0U;
               })
        .value_or(0U);
    if (trace && handle == 0) {
        static std::atomic<u32> traces{};
        if (traces.fetch_add(1, std::memory_order_relaxed) < 128) {
            const auto& code = info.handle_programs[program];
            LOG_WARNING(HW_GPU, "Handle program {} evaluated to 0: {} ({}, {:#x}); {} words: {:x}",
                        program, missing ? missing : "computed 0", missing_index, missing_offset,
                        code.size(), fmt::join(code.begin(), code.begin() + std::min<size_t>(code.size(), 48), " "));
        }
    }
    return handle;
}

} // namespace VideoCommon
