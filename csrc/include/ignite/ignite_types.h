#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>

using optim_sgd = torch::optim::SGD*;
using graph_function = torch::jit::GraphFunction*;
using script_module = torch::jit::script::Module*;
using stack = torch::jit::Stack*;

// casting to and from raw pointers
namespace make_raw {
    void* SGD(const optim_sgd& x);
    void* GraphFunction(const graph_function& x);
    void* ScriptModule(const script_module& x);
    void* Stack(const stack& x);
}

namespace from_raw {
    optim_sgd SGD(void* x);
    graph_function GraphFunction(void* x);
    script_module ScriptModule(void* x);
    stack Stack(void* x);
}
