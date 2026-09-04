#include "ax_engine_runner.hpp"

#include "ax_engine_api.h"
#include "ax_engine_type.h"
#include "ax_sys_api.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {
void check_ax(AX_S32 ret, const char* what) {
    if (ret != 0) throw std::runtime_error(std::string(what) + " failed");
}
}  // namespace

struct AxEngineRunner::Impl {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io{};
    std::vector<char> model;
    std::vector<std::string> in_names, out_names;
    std::unordered_map<std::string, size_t> in_idx, out_idx;
    bool engine_ready = false;
};

AxEngineRunner::AxEngineRunner(const std::string& model_path) : impl_(new Impl) {
    std::ifstream file(model_path.c_str(), std::ios::binary | std::ios::ate);
    if (!file) {
        delete impl_;
        throw std::runtime_error("cannot open axmodel: " + model_path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0);
    impl_->model.resize(static_cast<size_t>(size));
    file.read(impl_->model.data(), size);

    AX_ENGINE_NPU_ATTR_T attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

    try {
        // AX_SYS_Init must precede AX_ENGINE_Init or the latter returns an error.
        check_ax(AX_SYS_Init(), "AX_SYS_Init");
        check_ax(AX_ENGINE_Init(&attr), "AX_ENGINE_Init");
        impl_->engine_ready = true;

        check_ax(AX_ENGINE_CreateHandle(&impl_->handle, impl_->model.data(),
                                        static_cast<AX_U32>(impl_->model.size())),
                 "AX_ENGINE_CreateHandle");
        check_ax(AX_ENGINE_CreateContext(impl_->handle), "AX_ENGINE_CreateContext");
        check_ax(AX_ENGINE_GetIOInfo(impl_->handle, &impl_->info), "AX_ENGINE_GetIOInfo");

        const AX_U32 n_in = impl_->info->nInputSize;
        const AX_U32 n_out = impl_->info->nOutputSize;
        impl_->io.nInputSize = n_in;
        impl_->io.nOutputSize = n_out;
        impl_->io.pInputs = new AX_ENGINE_IO_BUFFER_T[n_in]();
        impl_->io.pOutputs = new AX_ENGINE_IO_BUFFER_T[n_out]();

        for (AX_U32 i = 0; i < n_in; ++i) {
            const AX_U32 bytes = impl_->info->pInputs[i].nSize;
            check_ax(AX_SYS_MemAllocCached(&impl_->io.pInputs[i].phyAddr,
                                           &impl_->io.pInputs[i].pVirAddr, bytes, 128,
                                           reinterpret_cast<const AX_S8*>("npu")),
                     "input alloc");
            impl_->io.pInputs[i].nSize = bytes;
            std::memset(impl_->io.pInputs[i].pVirAddr, 0, bytes);
            const std::string name = impl_->info->pInputs[i].pName;
            impl_->in_names.push_back(name);
            impl_->in_idx[name] = i;
        }
        for (AX_U32 i = 0; i < n_out; ++i) {
            const AX_U32 bytes = impl_->info->pOutputs[i].nSize;
            check_ax(AX_SYS_MemAllocCached(&impl_->io.pOutputs[i].phyAddr,
                                           &impl_->io.pOutputs[i].pVirAddr, bytes, 128,
                                           reinterpret_cast<const AX_S8*>("npu")),
                     "output alloc");
            impl_->io.pOutputs[i].nSize = bytes;
            const std::string name = impl_->info->pOutputs[i].pName;
            impl_->out_names.push_back(name);
            impl_->out_idx[name] = i;
        }
    } catch (...) {
        AxEngineRunner::Impl* dying = impl_;
        impl_ = nullptr;
        if (dying->handle) AX_ENGINE_DestroyHandle(dying->handle);
        if (dying->engine_ready) {
            AX_ENGINE_Deinit();
            AX_SYS_Deinit();
        }
        delete dying;
        throw;
    }
}

AxEngineRunner::~AxEngineRunner() {
    if (!impl_) return;
    if (impl_->io.pInputs) {
        for (AX_U32 i = 0; i < impl_->io.nInputSize; ++i)
            AX_SYS_MemFree(impl_->io.pInputs[i].phyAddr, impl_->io.pInputs[i].pVirAddr);
        delete[] impl_->io.pInputs;
    }
    if (impl_->io.pOutputs) {
        for (AX_U32 i = 0; i < impl_->io.nOutputSize; ++i)
            AX_SYS_MemFree(impl_->io.pOutputs[i].phyAddr, impl_->io.pOutputs[i].pVirAddr);
        delete[] impl_->io.pOutputs;
    }
    if (impl_->handle) AX_ENGINE_DestroyHandle(impl_->handle);
    if (impl_->engine_ready) {
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
    }
    delete impl_;
}

float* AxEngineRunner::input(const std::string& name) {
    auto it = impl_->in_idx.find(name);
    if (it == impl_->in_idx.end()) throw std::runtime_error("no such input: " + name);
    return static_cast<float*>(impl_->io.pInputs[it->second].pVirAddr);
}

const float* AxEngineRunner::output(const std::string& name) const {
    auto it = impl_->out_idx.find(name);
    if (it == impl_->out_idx.end()) throw std::runtime_error("no such output: " + name);
    return static_cast<const float*>(impl_->io.pOutputs[it->second].pVirAddr);
}

size_t AxEngineRunner::input_elems(const std::string& name) const {
    auto it = impl_->in_idx.find(name);
    if (it == impl_->in_idx.end()) throw std::runtime_error("no such input: " + name);
    return impl_->info->pInputs[it->second].nSize / sizeof(float);
}

size_t AxEngineRunner::output_elems(const std::string& name) const {
    auto it = impl_->out_idx.find(name);
    if (it == impl_->out_idx.end()) throw std::runtime_error("no such output: " + name);
    return impl_->info->pOutputs[it->second].nSize / sizeof(float);
}

bool AxEngineRunner::has_input(const std::string& name) const {
    return impl_->in_idx.count(name) != 0;
}

const std::vector<std::string>& AxEngineRunner::input_names() const { return impl_->in_names; }
const std::vector<std::string>& AxEngineRunner::output_names() const { return impl_->out_names; }

void AxEngineRunner::run() { check_ax(AX_ENGINE_RunSync(impl_->handle, &impl_->io), "AX_ENGINE_RunSync"); }
