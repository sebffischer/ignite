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
torch::Tensor ignite_run_script_module (script_module network, graph_function loss_fn, torch::Tensor input, torch::Tensor target, optim_sgd optimizer);
IGNITE_API void* _ignite_run_script_module (void* network, void* loss_fn, void* input, void* target, void* optimizer) {
  try {
    return  make_raw::Tensor(ignite_run_script_module(from_raw::ScriptModule(network), from_raw::GraphFunction(loss_fn), from_raw::Tensor(input), from_raw::Tensor(target), from_raw::SGD(optimizer)));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
std::vector<torch::Tensor> ignite_forward (torch::Tensor input, torch::Tensor weights, torch::Tensor bias, torch::Tensor old_h, torch::Tensor old_cell);
IGNITE_API void* _ignite_forward (void* input, void* weights, void* bias, void* old_h, void* old_cell) {
  try {
    return  make_raw::TensorList(ignite_forward(from_raw::Tensor(input), from_raw::Tensor(weights), from_raw::Tensor(bias), from_raw::Tensor(old_h), from_raw::Tensor(old_cell)));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
std::vector<torch::Tensor> ignite_backward (torch::Tensor grad_h, torch::Tensor grad_cell, torch::Tensor new_cell, torch::Tensor input_gate, torch::Tensor output_gate, torch::Tensor candidate_cell, torch::Tensor X, torch::Tensor gate_weights, torch::Tensor weights);
IGNITE_API void* _ignite_backward (void* grad_h, void* grad_cell, void* new_cell, void* input_gate, void* output_gate, void* candidate_cell, void* X, void* gate_weights, void* weights) {
  try {
    return  make_raw::TensorList(ignite_backward(from_raw::Tensor(grad_h), from_raw::Tensor(grad_cell), from_raw::Tensor(new_cell), from_raw::Tensor(input_gate), from_raw::Tensor(output_gate), from_raw::Tensor(candidate_cell), from_raw::Tensor(X), from_raw::Tensor(gate_weights), from_raw::Tensor(weights)));
  } IGNITE_HANDLE_EXCEPTION
  return (void*) NULL;
}
void delete_optim_sgd (void* x);
IGNITE_API void _delete_optim_sgd (void* x) {
  try {
     (delete_optim_sgd(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_graph_function (void* x);
IGNITE_API void _delete_graph_function (void* x) {
  try {
     (delete_graph_function(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_script_module2 (void* x);
IGNITE_API void _delete_script_module2 (void* x) {
  try {
     (delete_script_module2(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
void delete_stack2 (void* x);
IGNITE_API void _delete_stack2 (void* x) {
  try {
     (delete_stack2(x));
  } IGNITE_HANDLE_EXCEPTION
  
}
