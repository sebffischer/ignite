#include <torch/torch.h>
#include "ignite/ignite_types.h"
#include <lantern/types.h>


// When making a raw pointer we distinguish two cases:
// 1. The object x is managing its own memory.
//    The cases for this are:
//    - The optimizer, which are heap-allocated by the ignite functions in ignite.cpp
//    - A shared pointer (such as a torch::Tensor) is returned
//    - A custom heap-allocated object is returned (e.g. the adamw_options)
// 2. The object x is not managing its own memory.
//    This is needed for things like the param groups or optimizer options where x is only
//    a pointer and the memory is managed by the optimizer.
//    We never directly return such objects to R but only use it to communicate it to Rcpp.
//    For those, we also must not register the destructor on the Rcpp side as this might
//    otherwise free the memory twice.

namespace make_raw {

void* Optim(const optim& x) {
    return x;
}
void* SGD(const optim_sgd& x) {
    return x;
}
void* Adam(const optim_adam& x) {
    return x;
}
void* AdamW(const optim_adamw& x) {
    return x;
}
void* Adagrad(const optim_adagrad& x) {
    return x;
}
void* RMSprop(const optim_rmsprop& x) {
    return x;
}
void* ScriptModule(const script_module& x) {
    return x;
}
void* TorchStack(const torch_stack& x) {
    return x;
}
void* OptimParamGroups(const optim_param_groups& x) {
    return make_ptr<optim_param_groups>(x);
}
void* OptimParamGroup(const optim_param_group& x) {
    return x;
}
void* AdamWParamGroup(const adamw_param_group& x) {
    return x;
}
void* AdamWParamGroups(const adamw_param_groups& x) {
    // TODO: I think we need to access the groups here
    return make_ptr<adamw_param_groups>(x);
}
void* AdamWStates(const adamw_states& x) {
    return make_ptr<adamw_states>(x);
}
void* AdamWState(const adamw_state& x) {
    return x;
}
void* AdamWOptions(const adamw_options& x) {
    return make_ptr<adamw_options>(x);
}
void* OptimOptions(const optim_options& x) {
    return x;
}
void* StringVector(const string_vector& x) {
    return make_ptr<string_vector>(x);
}
}

namespace from_raw {
optim Optim(void* x) {
    return reinterpret_cast<optim>(x);
}
optim_sgd SGD(void* x) {
    return reinterpret_cast<optim_sgd>(x);
}
optim_adam Adam(void* x) {
    return reinterpret_cast<optim_adam>(x);
}
optim_adamw AdamW(void* x) {
    return reinterpret_cast<optim_adamw>(x);
}
optim_adagrad Adagrad(void* x) {
    return *reinterpret_cast<optim_adagrad*>(x);
}
optim_rmsprop RMSprop(void* x) {
    return *reinterpret_cast<optim_rmsprop*>(x);
}
script_module ScriptModule(void* x) {
    return *reinterpret_cast<script_module*>(x);
}
torch_stack TorchStack(void* x) {
    return reinterpret_cast<torch_stack>(x);
}
optim_param_groups OptimParamGroups(void* x) {
    return *reinterpret_cast<optim_param_groups*>(x);
}
optim_param_group OptimParamGroup(void* x) {
    return *reinterpret_cast<optim_param_group*>(x);
}
adamw_states AdamWStates(void* x) {
    return *reinterpret_cast<adamw_states*>(x);
}
adamw_state AdamWState(void* x) {
    return *reinterpret_cast<adamw_state*>(x);
}
adamw_options AdamWOptions(void* x) {
    return *reinterpret_cast<adamw_options*>(x);
}
optim_options OptimOptions(void* x) {
    return *reinterpret_cast<optim_options*>(x);
}
string_vector StringVector(void* x) {
    return *reinterpret_cast<string_vector*>(x);
}
}

// TODO: Remove the deleters that are not needed

// [[torch::export]]
void delete_optim(void* x) {
  delete reinterpret_cast<optim>(x);
}
// [[torch::export]]
void delete_optim_sgd(void* x) {
  delete reinterpret_cast<optim_sgd>(x);
}
// [[torch::export]]
void delete_optim_adam(void* x) {
  delete reinterpret_cast<optim_adam>(x);
}
// [[torch::export]]
void delete_optim_adamw(void* x) {
  delete reinterpret_cast<optim_adamw>(x);
}
// [[torch::export]]
void delete_optim_adagrad(void* x) {
  delete reinterpret_cast<optim_adagrad>(x);
}
// [[torch::export]]
void delete_optim_rmsprop(void* x) {
  delete reinterpret_cast<optim_rmsprop>(x);
}
// [[torch::export]]
void delete_optim_param_groups(void* x) {
  delete reinterpret_cast<optim_param_groups*>(x);
}
// [[torch::export]]
void delete_optim_param_group(void* x) {
  delete reinterpret_cast<optim_param_group*>(x);
}
// [[torch::export]]
void delete_adamw_param_groups(void* x) {
  delete reinterpret_cast<adamw_param_groups*>(x);
}
// [[torch::export]]
void delete_adamw_param_group(void* x) {
  delete reinterpret_cast<adamw_param_group*>(x);
}

// [[torch::export]]
void delete_adamw_states(void* x) {
  delete reinterpret_cast<adamw_states*>(x);
}

// [[torch::export]]
void delete_adamw_state(void* x) {
  delete reinterpret_cast<adamw_state>(x);
}

// [[torch::export]]
void delete_adamw_options(void* x) {
  delete reinterpret_cast<adamw_options*>(x);
}
