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

optim_param_groups ignite_get_param_groups(optim opt) {
  // this works for all optimizers.
  // However in Rcpp -> SEXP conversion we then iterate over the void*s and call a optimizer-specific getter
  std::vector<optim_param_group> param_groups;
  // we convert to a vector of pointers so we can iterate over it on the Rcpp side
  // (we know the size of void pointers)
  for (torch::optim::OptimizerParamGroup& group : opt->param_groups()) {
    param_groups.push_back(&group);
  }

  return param_groups;
}

// [[torch::export(register_types=c("adamw_param_groups", "AdamWParamGroups", "void*", "ignite::adamw_param_groups"))]]
adamw_param_groups ignite_adamw_get_param_groups(optim_adamw groups) {
  std::cout << "ignite_adamw_get_param_groups" << std::endl;
  adamw_param_groups param_groups = { ignite_get_param_groups(groups) };
  return param_groups;
}

// [[torch::export]]
int ignite_adamw_param_groups_size(optim_adamw opt) {
  return opt->param_groups().size();
}

// [[torch::export(register_types=c("optim_param_group", "OptimParamGroup", "void*", "ignite::optim_param_group"))]]
std::vector<torch::Tensor> ignite_optim_get_param_group_params(optim_param_group group) {
  auto pars = group->params();
  // print dim of pars
  std::cout << "dimension of the params: " << pars[0].dim() << std::endl;

  return pars;
}

// [[torch::export(register_types=list(c("adamw_options", "AdamWOptions", "void*", "ignite::adamw_options")))]]
adamw_options ignite_adamw_get_param_group_options(optim_param_group group) {
  auto& x  = static_cast<torch::optim::AdamWOptions&>(group->options());
  return adamw_options{x.lr(), x.weight_decay(), x.betas(), x.eps(), x.amsgrad()};
}

// [[torch::export]]
void ignite_adamw_set_param_group_options(optim_adamw opt, int i, adamw_options options) {
  // get the i-th param group and set it
  auto& param_group = (opt->param_groups())[i];

  auto& x = static_cast<torch::optim::AdamWOptions&>(param_group.options());
  x.lr(options.lr);
  x.weight_decay(options.weight_decay);
  x.betas(options.betas);
  x.eps(options.eps);
  x.amsgrad(options.amsgrad);
}


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

// [[torch::export(register_types=list(c("adamw_states", "AdamWStates", "void*", "ignite::adamw_states"), c("adamw_state", "AdamWState", "void*", "ignite::adamw_state")))]]
adamw_states ignite_adamw_get_states(optim_adamw opt) {
  adamw_states states;

  // Collect keys in the order of param groups
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
      auto state_it = opt->state().find(key);
      // TODO: Check whether this actually does what we want
      if (state_it != opt->state().end()) {
        auto* adamw_state = static_cast<torch::optim::AdamWParamState*>(state_it->second.get());
        states.push_back(adamw_state);
      } else {
        // runtime error
        throw std::runtime_error("State not found");
      }
    }
  }

  return states;
}

void ignite_adamw_set_states(optim_adamw opt, adamw_states states) {
  // Check that lengths are the same
  if (opt->state().size() != states.size()) {
    throw std::runtime_error("State lengths are different");
  }

  size_t i = 0;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
      auto state_it = opt->state().find(key);
      // TODO: Check whether this actually does what we want
      if (state_it != opt->state().end()) {
        auto* current_state = static_cast<torch::optim::AdamWParamState*>(state_it->second.get());
        current_state->exp_avg(states[i]->exp_avg());
        current_state->exp_avg_sq(states[i]->exp_avg_sq());
        current_state->max_exp_avg_sq(states[i]->max_exp_avg_sq());
        auto step = states[i]->step();
        // convert step from torch::kLong to int64_t
        current_state->step(static_cast<int64_t>(step));
      } else {
        // runtime error
        throw std::runtime_error("State not found");
      }
    }
    ++i;
  }
}

// FIXME: We can just return a list of tensors or a tuple?

// [[torch::export]]
std::vector<torch::Tensor> ignite_adamw_get_state(adamw_state state) {
  auto exp_avg   = state->exp_avg();
  auto exp_avg_sq = state->exp_avg_sq();
  auto max_exp_avg_sq = state->max_exp_avg_sq();
  auto step = torch::scalar_tensor(state->step(), torch::kLong);

  return {exp_avg, exp_avg_sq, max_exp_avg_sq, step};
}

// [[torch::export]]
torch::Tensor adamw_state_exp_avg(adamw_state state) {
  auto x = state->exp_avg();
  return x;
}
// [[torch::export]]
torch::Tensor adamw_state_exp_avg_sq(adamw_state state) {
  auto x = state->exp_avg();
  return x;
}
// [[torch::export]]
torch::Tensor adamw_state_max_exp_avg_sq(adamw_state state) {
  auto x = state->max_exp_avg_sq();
  return x;
}
// [[torch::export]]
torch::Tensor adamw_state_step(adamw_state state) {
  auto step = state->step();
  // step is a long long, but R ints are 32 bit, so we represent it as a torch tensor which allows
  // for higher precision
  // this is also consistent with how the R adamw returns the step
  auto step_tensor = torch::scalar_tensor(step, torch::kLong);
  return step_tensor;
}

// [[torch::export(register_types=list(c("script_module", "ScriptModule", "void*", "Rcpp::XPtr<XPtrTorchScriptModule>"), c("torch_stack", "TorchStack", "void*", "XPtrTorchStack"), c("optim", "Optim", "void*", "ignite::optim")))]]
std::vector<torch::Tensor> ignite_opt_step(script_module network, script_module loss_fn, torch_stack input, torch::Tensor target, optim optimizer) {
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

