#include <ATen/core/stack.h>
#include <ATen/ops/_batch_norm_impl_index.h>
#include <ATen/ops/randn.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#define LANTERN_TYPES_IMPL // Should be defined only in a single file.
#include <lantern/types.h>
#include <vector>
#include "ignite/ignite.h"
#include "ignite/ignite_types.h"

#include <torch/script.h>  // One-stop header.


// this is working for all optimizers

// [[torch::export(register_types=c("adamw_param_groups", "AdamWParamGroups", "void*", "ignite::adamw_param_groups"))]]
adamw_param_groups ignite_adamw_get_param_groups(optim_adamw groups) {
  // param_groups returns a reference to a vector so it's still the obligation of the optimizer
  // to manage the memory
  return adamw_param_groups { &groups->param_groups() };
}
//
// [[torch::export]]
int ignite_adamw_param_groups_size(optim_param_groups groups) {
  // this is the same method for all optimizers
  return groups->size();
}

// [[torch::export(register_types=list(c("optim_param_group", "OptimParamGroup", "void*", "ignite::optim_param_group"), c("optim_param_groups", "OptimParamGroups", "void*", "ignite::optim_param_groups")))]]
std::vector<torch::Tensor> ignite_optim_get_param_group_params(optim_param_groups groups, int i) {
  auto group = (*groups)[i];
  return group.params();
}


// We don't need to call this from R as it is just used on the Rcpp side
// [[torch::export(rcpp = FALSE, register_types=list(c("adamw_options", "AdamWOptions", "adamw_options", "adamw_options")))]]
adamw_options ignite_adamw_get_param_group_options(optim_param_groups groups, int i) {
  auto group = (*groups)[i];
  auto& x = static_cast<torch::optim::AdamWOptions&>(group.options());
  auto betas = x.betas();
  adamw_options opts;
  opts.lr = x.lr();
  opts.weight_decay = x.weight_decay();
  opts.betas[0] = std::get<0>(betas);
  opts.betas[1] = std::get<1>(betas);
  opts.eps = x.eps();
  opts.amsgrad = x.amsgrad();
  return opts;
}

// [[torch::export(rcpp = FALSE)]]
void ignite_adamw_set_param_group_options(optim_adamw opt, int i, adamw_options options) {
  auto& group = (opt->param_groups())[i];
  auto& options_ref = group.options();
  auto& x = reinterpret_cast<torch::optim::AdamWOptions&>(options_ref);
  x.lr(options.lr);
  x.weight_decay(options.weight_decay);
  x.betas(std::make_tuple(options.betas[0], options.betas[1]));
  x.eps(options.eps);
  x.amsgrad(options.amsgrad);
  return;
}

//void ignite_adamw_set_param_group_options(optim_adamw opt, int i, adamw_options options) {
//  // get the i-th param group and set it
//  auto& param_group = (opt->param_groups())[i];
//
//  auto& x = static_cast<torch::optim::AdamWOptions&>(param_group.options());
//  x.lr(options.lr);
//  x.weight_decay(options.weight_decay);
//  x.betas(std::make_tuple(options.betas[0], options.betas[1]));
//  x.eps(options.eps);
//  x.amsgrad(options.amsgrad);
//}


//sgd_param_groups ignite_sgd_get_param_groups(optim_sgd opt) {
//  // iterate over the param groups and call ignite_sgd_get_param_group for each one and push the results into a vector
//  sgd_param_groups param_groups;
//  for (torch::optim::OptimizerParamGroup& group : opt->param_groups()) {
//    param_groups.push_back(sgd_param_group(group));
//  }
//  return param_groups;
//}
//
//void ignite_sgd_set_param_groups(optim_sgd opt, sgd_param_groups param_groups) {
//  if (opt->param_groups().size() != param_groups.size()) {
//    throw std::runtime_error("Parameter groups have different lengths");
//  }
//
//  // zip the param_groups and opt->param_groups by iterating over the indices
//  for (size_t i = 0; i < opt->param_groups().size(); ++i) {
//    auto& opt_group = opt->param_groups()[g];
//    auto& param_group = param_groups[i];
//    // TODO check that the params all point to the same tensors
//    opt_group.set_options(std::make_unique<torch::optim::SGDOptions>(param_group.to_sgd_options()));
//  }
//}

//void ignite_adamw_set_param_groups(optim_adamw opt, adamw_param_groups param_groups) {
//  if (opt->param_groups().size() != param_groups.size()) {
//    throw std::runtime_error("Parameter groups have different lengths");
//  }
//  // zip the param_groups and opt->param_groups by iterating over the indices
//  for (size_t i = 0; i < opt->param_groups().size(); ++i) {
//    auto& opt_group = opt->param_groups()[i];
//    auto& param_group = param_groups[i];
//    opt_group.set_options(std::make_unique<torch::optim::AdamWOptions>(param_group.to_adamw_options()));
//  }
//}

// [[torch::export]]
std::vector<torch::Tensor> ignite_adamw_get_states(optim_adamw opt) {
  // here we iterate over the param states in the order of param groups
  // and cast those param states to the optimizer-specific type
  // the states themselves are still owned by the optimizer itself
  // however, the newly heap-allocated vector is owned by this function
  // so we will have to delete it later

  // Create an empty vector to store the AdamW parameter states

  std::vector<torch::Tensor> tensors;

  // Iterate through each parameter group in the optimizer
  for (const auto& group : opt->param_groups()) {
    // For each parameter in the group
    for (const auto& param : group.params()) {
      // Get a unique string key for this parameter tensor
      // The key is created from the tensor's implementation pointer
      auto key = c10::guts::to_string(param.unsafeGetTensorImpl());

      // Look up this parameter's state in the optimizer's state map
      // The state map is a flat_hash_map that maps parameter keys to their optimizer states
      auto state_it = opt->state().find(key);

      if (state_it != opt->state().end()) {
        // If state exists for this parameter:
        // 1. Get raw pointer to the OptimizerParamState from the unique_ptr
        // 2. Cast it to AdamWParamState since we know this is an AdamW optimizer
        // 3. Store the pointer in our states vector
        auto base_state = state_it->second.get(); // Get raw pointer from unique_ptr
        auto adamw_state = static_cast<torch::optim::AdamWParamState*>(base_state);
        // we need to clone because the tensors are behind unique pointers
        // but we want ownership
        tensors.push_back(adamw_state->exp_avg().clone());
        tensors.push_back(adamw_state->exp_avg_sq().clone());
        tensors.push_back(adamw_state->max_exp_avg_sq().clone());
        tensors.push_back(torch::scalar_tensor(adamw_state->step(), torch::kLong));
      } else {
        // This parameter should have state - error if not found
        throw std::runtime_error("State not found for parameter");
      }
    }
  }
  std::cout << "tensors size: " << tensors.size() << std::endl;
  return tensors;
}

// [[torch::export]]
void ignite_adamw_set_states(optim_adamw opt, torch::TensorList states) {
  size_t i = 0;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
      auto state_it = opt->state().find(key);
      // TODO: Check whether this actually does what we want
      if (state_it != opt->state().end()) {
        auto* current_state = static_cast<torch::optim::AdamWParamState*>(state_it->second.get());
        current_state->exp_avg(states[i]);
        current_state->exp_avg_sq(states[i + 1]);
        current_state->max_exp_avg_sq(states[i + 2]);
        auto step = states[i + 3];
        // convert step from torch::kLong to int64_t
        current_state->step(step.item<int64_t>());
      } else {
        // runtime error
        throw std::runtime_error("State not found");
      }
    }
    i += 4;
  }
}


// [[torch::export(register_types=list(c("script_module", "ScriptModule", "void*", "Rcpp::XPtr<XPtrTorchScriptModule>"), c("torch_stack", "TorchStack", "void*", "XPtrTorchStack"), c("optim", "Optim", "void*", "ignite::optim")))]]
std::vector<torch::Tensor> ignite_opt_step(script_module network, script_module loss_fn, torch_stack input, torch::Tensor target, optim_adamw optimizer) {
  // TODO: optim_adamw -> optim
  optimizer->zero_grad();

  auto out = (*network)(*input);
  auto loss_inputs = new torch::jit::Stack();
  loss_inputs->push_back(out);
  loss_inputs->push_back(target);

  auto loss = (*loss_fn)(*loss_inputs);
  loss.toTensor().backward();
  optimizer->step();

  std::vector<torch::Tensor> result;
  result.push_back(loss.toTensor());
  result.push_back(out.toTensor());
  return result;
}

// [[torch::export]]
torch::Tensor ignite_predict_step(script_module network, torch_stack input) {
  torch::NoGradGuard no_grad;
  auto out = (*network)(*input);
  return out.toTensor();
}

// TODO: Add betas as a tuple
// [[torch::export(register_types=c("optim_adamw", "AdamW", "void*", "ignite::optim_adamw"))]]
optim_adamw ignite_adamw(torch::TensorList params, double lr, double beta1, double beta2,
                        double eps, double weight_decay, bool amsgrad) {

  auto options = torch::optim::AdamWOptions(lr)
    .betas(std::make_tuple(beta1, beta2))
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);

  return new torch::optim::AdamW(params.vec(), options);
}

// [[torch::export]]
void ignite_adamw_step(optim_adamw opt) {
  opt->step();
}

// [[torch::export]]
void ignite_adamw_zero_grad(optim_adamw opt) {
  opt->zero_grad();
}


IGNITE_API int _raise_exception ()
{
  try {
    throw std::runtime_error("Error from IGNITE");
  } IGNITE_HANDLE_EXCEPTION
  return 1;
}

