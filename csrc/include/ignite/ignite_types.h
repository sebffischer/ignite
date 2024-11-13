#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#include <vector>

using optim = torch::optim::Optimizer*;
using optim_sgd = torch::optim::SGD*;
using optim_adam = torch::optim::Adam*;
using optim_adamw = torch::optim::AdamW*;
using optim_adagrad = torch::optim::Adagrad*;
using optim_rmsprop = torch::optim::RMSprop*;
using script_module = torch::jit::script::Module*;
using torch_stack = torch::jit::Stack*;

// it is a vector so on the Rcpp side we can cast it to a std::vector<void*> and iterate over it.
using adamw_state = torch::optim::AdamWParamState*;
using adamw_states = std::vector<adamw_state>;
using string_vector = std::vector<std::string>;

// To implement param_groups for the R optimizer, we work with OptimizerParamGroup,
// which gives direct access to the parameters.
// To get access to the optimizer options (lr etc.) we need to downcast to the specific optimizer type.

using optim_param_group = torch::optim::OptimizerParamGroup*;
using optim_param_groups = std::vector<optim_param_group>;
using optim_options = torch::optim::OptimizerOptions*;

// even though all optimizers store their param groups in a std::vector<torch::optim::OptimizerParamGroup*>,
// we need different types for the different optimizers because we need to know from the type
// How to convert it to the Rcpp type.

struct adamw_param_groups {
    optim_param_groups groups;
};

struct adamw_options {
    double lr;
    double weight_decay;
    std::tuple<double, double> betas;
    double eps;
    bool amsgrad;
};

// TODO: This is wrong
using adamw_param_group = torch::optim::AdamWOptions*;

namespace make_raw {
    void* Optim(const optim& x);
    void* SGD(const optim_sgd& x);
    void* Adam(const optim_adam& x);
    void* AdamW(const optim_adamw& x);
    void* Adagrad(const optim_adagrad& x);
    void* RMSprop(const optim_rmsprop& x);
    void* ScriptModule(const script_module& x);
    void* TorchStack(const torch_stack& x);
    void* OptimParamGroups(const optim_param_groups& x);
    void* OptimParamGroup(const optim_param_group& x);
    void* AdamWParamGroups(const adamw_param_groups& x);
    void* AdamWStates(const adamw_states& x);
    void* AdamWState(const adamw_state& x);
    void* AdamWOptions(const adamw_options& x);
    void* OptimOptions(const optim_options& x);
    void* StringVector(const string_vector& x);
}

namespace from_raw {
    optim Optim(void* x);
    optim_sgd SGD(void* x);
    optim_adam Adam(void* x);
    optim_adamw AdamW(void* x);
    optim_adagrad Adagrad(void* x);
    optim_rmsprop RMSprop(void* x);
    script_module ScriptModule(void* x);
    torch_stack TorchStack(void* x);
    optim_param_groups OptimParamGroups(void* x);
    optim_param_group OptimParamGroup(void* x);
    adamw_param_groups AdamWParamGroups(void* x);
    adamw_param_group AdamWParamGroup(void* x);
    adamw_states AdamWStates(void* x);
    adamw_state AdamWState(void* x);
    adamw_options AdamWOptions(void* x);
    optim_options OptimOptions(void* x);
    string_vector StringVector(void* x);
}
