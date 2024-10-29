#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>

using optim_sgd = torch::optim::SGD*;
using optim_adam = torch::optim::Adam*;
using optim_adamw = torch::optim::AdamW*;
using optim_adagrad = torch::optim::Adagrad*;
using optim_rmsprop = torch::optim::RMSprop*;
using script_module = torch::jit::script::Module*;
using torch_stack = torch::jit::Stack*;

// casting to and from raw pointers
namespace make_raw {
    void* SGD(const optim_sgd& x);
    void* Adam(const optim_adam& x);
    void* AdamW(const optim_adamw& x);
    void* Adagrad(const optim_adagrad& x);
    void* RMSprop(const optim_rmsprop& x);
    void* ScriptModule(const script_module& x);
    void* TorchStack(const torch_stack& x);
}

namespace from_raw {
    optim_sgd SGD(void* x);
    optim_adam Adam(void* x);
    optim_adamw AdamW(void* x);
    optim_adagrad Adagrad(void* x);
    optim_rmsprop RMSprop(void* x);
    script_module ScriptModule(void* x);
    torch_stack TorchStack(void* x);
}
