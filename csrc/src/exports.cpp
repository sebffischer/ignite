// Generated by using torchexport::export() -> do not edit by hand
#include "ignite/ignite.h"
#include <lantern/types.h>
#include "ignite/ignite_types.h"
void * p_ignite_last_error = NULL;

IGNITE_API void* ignite_last_error()
{
  return p_ignite_last_error;
}

IGNITE_API void ignite_last_error_clear()
{
  p_ignite_last_error = NULL;
}

optim_sgd ignite_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
IGNITE_API void* _ignite_sgd (void* params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  try {
    return  make_raw::SGD(ignite_sgd(from_raw::TensorList(params), lr, momentum, dampening, weight_decay, nesterov));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void ignite_sgd_step (optim_sgd opt);
IGNITE_API void _ignite_sgd_step (void* opt) {
  try {
     (ignite_sgd_step(from_raw::SGD(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
void ignite_sgd_zero_grad (optim_sgd opt);
IGNITE_API void _ignite_sgd_zero_grad (void* opt) {
  try {
     (ignite_sgd_zero_grad(from_raw::SGD(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
optim_adam ignite_adam (torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
IGNITE_API void* _ignite_adam (void* params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad) {
  try {
    return  make_raw::Adam(ignite_adam(from_raw::TensorList(params), lr, beta1, beta2, eps, weight_decay, amsgrad));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void ignite_adam_step (optim_adam opt);
IGNITE_API void _ignite_adam_step (void* opt) {
  try {
     (ignite_adam_step(from_raw::Adam(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
void ignite_adam_zero_grad (optim_adam opt);
IGNITE_API void _ignite_adam_zero_grad (void* opt) {
  try {
     (ignite_adam_zero_grad(from_raw::Adam(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
optim_adamw ignite_adamw (torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
IGNITE_API void* _ignite_adamw (void* params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad) {
  try {
    return  make_raw::AdamW(ignite_adamw(from_raw::TensorList(params), lr, beta1, beta2, eps, weight_decay, amsgrad));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void ignite_adamw_step (optim_adamw opt);
IGNITE_API void _ignite_adamw_step (void* opt) {
  try {
     (ignite_adamw_step(from_raw::AdamW(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
void ignite_adamw_zero_grad (optim_adamw opt);
IGNITE_API void _ignite_adamw_zero_grad (void* opt) {
  try {
     (ignite_adamw_zero_grad(from_raw::AdamW(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
optim_adagrad ignite_adagrad (torch::TensorList params, double lr, double lr_decay, double weight_decay, double initial_accumulator_value, double eps);
IGNITE_API void* _ignite_adagrad (void* params, double lr, double lr_decay, double weight_decay, double initial_accumulator_value, double eps) {
  try {
    return  make_raw::Adagrad(ignite_adagrad(from_raw::TensorList(params), lr, lr_decay, weight_decay, initial_accumulator_value, eps));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void ignite_adagrad_step (optim_adagrad opt);
IGNITE_API void _ignite_adagrad_step (void* opt) {
  try {
     (ignite_adagrad_step(from_raw::Adagrad(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
void ignite_adagrad_zero_grad (optim_adagrad opt);
IGNITE_API void _ignite_adagrad_zero_grad (void* opt) {
  try {
     (ignite_adagrad_zero_grad(from_raw::Adagrad(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
optim_rmsprop ignite_rmsprop (torch::TensorList params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered);
IGNITE_API void* _ignite_rmsprop (void* params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered) {
  try {
    return  make_raw::RMSprop(ignite_rmsprop(from_raw::TensorList(params), lr, alpha, eps, weight_decay, momentum, centered));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void ignite_rmsprop_step (optim_rmsprop opt);
IGNITE_API void _ignite_rmsprop_step (void* opt) {
  try {
     (ignite_rmsprop_step(from_raw::RMSprop(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
void ignite_rmsprop_zero_grad (optim_rmsprop opt);
IGNITE_API void _ignite_rmsprop_zero_grad (void* opt) {
  try {
     (ignite_rmsprop_zero_grad(from_raw::RMSprop(opt)));
  } IGNITE_HANDLE_EXCEPTION
  
}
std::vector<torch::Tensor> ignite_opt_step (script_module network, script_module loss_fn, torch_stack input, torch::Tensor target, optim_sgd optimizer);
IGNITE_API void* _ignite_opt_step (void* network, void* loss_fn, void* input, void* target, void* optimizer) {
  try {
    return  make_raw::TensorList(ignite_opt_step(from_raw::ScriptModule(network), from_raw::ScriptModule(loss_fn), from_raw::TorchStack(input), from_raw::Tensor(target), from_raw::SGD(optimizer)));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
torch::Tensor ignite_predict_step (script_module network, torch_stack input);
IGNITE_API void* _ignite_predict_step (void* network, void* input) {
  try {
    return  make_raw::Tensor(ignite_predict_step(from_raw::ScriptModule(network), from_raw::TorchStack(input)));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void delete_optim_sgd (void* x);
IGNITE_API void _delete_optim_sgd (void* x) {
  try {
     (delete_optim_sgd(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_optim_adam (void* x);
IGNITE_API void _delete_optim_adam (void* x) {
  try {
     (delete_optim_adam(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_optim_adamw (void* x);
IGNITE_API void _delete_optim_adamw (void* x) {
  try {
     (delete_optim_adamw(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_optim_adagrad (void* x);
IGNITE_API void _delete_optim_adagrad (void* x) {
  try {
     (delete_optim_adagrad(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_optim_rmsprop (void* x);
IGNITE_API void _delete_optim_rmsprop (void* x) {
  try {
     (delete_optim_rmsprop(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
