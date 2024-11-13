// Generated by using torchexport::export() -> do not edit by hand
#pragma once

#ifdef _WIN32
#ifndef IGNITE_HEADERS_ONLY
#define IGNITE_API extern "C" __declspec(dllexport)
#else
#define IGNITE_API extern "C" __declspec(dllimport)
#endif
#else
#define IGNITE_API extern "C"
#endif

#ifndef IGNITE_HANDLE_EXCEPTION
#define IGNITE_HANDLE_EXCEPTION                                  \
catch(const std::exception& ex) {                                  \
  p_ignite_last_error = make_raw::string(ex.what());             \
} catch (std::string& ex) {                                        \
  p_ignite_last_error = make_raw::string(ex);                    \
} catch (...) {                                                    \
  p_ignite_last_error = make_raw::string("Unknown error. ");     \
}
#endif

void host_exception_handler ();
extern void* p_ignite_last_error;
IGNITE_API void* ignite_last_error ();
IGNITE_API void ignite_last_error_clear();

IGNITE_API void* _ignite_adamw_get_param_groups (void* groups);
IGNITE_API void* _ignite_optim_get_param_group_params (void* group);
IGNITE_API void* _ignite_adamw_get_param_group_options (void* group);
IGNITE_API void _ignite_adamw_set_param_group_options (void* opt, int i, void* options);
IGNITE_API void* _ignite_adamw_get_states (void* opt);
IGNITE_API void* _ignite_adamw_get_state (void* state);
IGNITE_API void* _adamw_state_exp_avg (void* state);
IGNITE_API void* _adamw_state_exp_avg_sq (void* state);
IGNITE_API void* _adamw_state_max_exp_avg_sq (void* state);
IGNITE_API void* _adamw_state_step (void* state);
IGNITE_API void* _ignite_opt_step (void* network, void* loss_fn, void* input, void* target, void* optimizer);
IGNITE_API void* _ignite_predict_step (void* network, void* input);
IGNITE_API void* _ignite_adamw (void* params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
IGNITE_API void _ignite_adamw_step (void* opt);
IGNITE_API void _ignite_adamw_zero_grad (void* opt);
IGNITE_API void _delete_optim (void* x);
IGNITE_API void _delete_optim_sgd (void* x);
IGNITE_API void _delete_optim_adam (void* x);
IGNITE_API void _delete_optim_adamw (void* x);
IGNITE_API void _delete_optim_adagrad (void* x);
IGNITE_API void _delete_optim_rmsprop (void* x);
IGNITE_API void _delete_optim_param_groups (void* x);
IGNITE_API void _delete_optim_param_group (void* x);
IGNITE_API void _delete_adamw_param_groups (void* x);
IGNITE_API void _delete_adamw_param_group (void* x);
IGNITE_API void _delete_adamw_states (void* x);
IGNITE_API void _delete_adamw_state (void* x);
IGNITE_API void _delete_adamw_options (void* x);

#ifdef RCPP_VERSION
inline void* ignite_adamw_get_param_groups (void* groups) {
  auto ret =  _ignite_adamw_get_param_groups(groups);
  host_exception_handler();
  return ret;
}
inline void* ignite_optim_get_param_group_params (void* group) {
  auto ret =  _ignite_optim_get_param_group_params(group);
  host_exception_handler();
  return ret;
}
inline void* ignite_adamw_get_param_group_options (void* group) {
  auto ret =  _ignite_adamw_get_param_group_options(group);
  host_exception_handler();
  return ret;
}
inline void ignite_adamw_set_param_group_options (void* opt, int i, void* options) {
   _ignite_adamw_set_param_group_options(opt, i, options);
  host_exception_handler();
  
}
inline void* ignite_adamw_get_states (void* opt) {
  auto ret =  _ignite_adamw_get_states(opt);
  host_exception_handler();
  return ret;
}
inline void* ignite_adamw_get_state (void* state) {
  auto ret =  _ignite_adamw_get_state(state);
  host_exception_handler();
  return ret;
}
inline void* adamw_state_exp_avg (void* state) {
  auto ret =  _adamw_state_exp_avg(state);
  host_exception_handler();
  return ret;
}
inline void* adamw_state_exp_avg_sq (void* state) {
  auto ret =  _adamw_state_exp_avg_sq(state);
  host_exception_handler();
  return ret;
}
inline void* adamw_state_max_exp_avg_sq (void* state) {
  auto ret =  _adamw_state_max_exp_avg_sq(state);
  host_exception_handler();
  return ret;
}
inline void* adamw_state_step (void* state) {
  auto ret =  _adamw_state_step(state);
  host_exception_handler();
  return ret;
}
inline void* ignite_opt_step (void* network, void* loss_fn, void* input, void* target, void* optimizer) {
  auto ret =  _ignite_opt_step(network, loss_fn, input, target, optimizer);
  host_exception_handler();
  return ret;
}
inline void* ignite_predict_step (void* network, void* input) {
  auto ret =  _ignite_predict_step(network, input);
  host_exception_handler();
  return ret;
}
inline void* ignite_adamw (void* params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad) {
  auto ret =  _ignite_adamw(params, lr, beta1, beta2, eps, weight_decay, amsgrad);
  host_exception_handler();
  return ret;
}
inline void ignite_adamw_step (void* opt) {
   _ignite_adamw_step(opt);
  host_exception_handler();
  
}
inline void ignite_adamw_zero_grad (void* opt) {
   _ignite_adamw_zero_grad(opt);
  host_exception_handler();
  
}
inline void delete_optim (void* x) {
   _delete_optim(x);
  host_exception_handler();
  
}
inline void delete_optim_sgd (void* x) {
   _delete_optim_sgd(x);
  host_exception_handler();
  
}
inline void delete_optim_adam (void* x) {
   _delete_optim_adam(x);
  host_exception_handler();
  
}
inline void delete_optim_adamw (void* x) {
   _delete_optim_adamw(x);
  host_exception_handler();
  
}
inline void delete_optim_adagrad (void* x) {
   _delete_optim_adagrad(x);
  host_exception_handler();
  
}
inline void delete_optim_rmsprop (void* x) {
   _delete_optim_rmsprop(x);
  host_exception_handler();
  
}
inline void delete_optim_param_groups (void* x) {
   _delete_optim_param_groups(x);
  host_exception_handler();
  
}
inline void delete_optim_param_group (void* x) {
   _delete_optim_param_group(x);
  host_exception_handler();
  
}
inline void delete_adamw_param_groups (void* x) {
   _delete_adamw_param_groups(x);
  host_exception_handler();
  
}
inline void delete_adamw_param_group (void* x) {
   _delete_adamw_param_group(x);
  host_exception_handler();
  
}
inline void delete_adamw_states (void* x) {
   _delete_adamw_states(x);
  host_exception_handler();
  
}
inline void delete_adamw_state (void* x) {
   _delete_adamw_state(x);
  host_exception_handler();
  
}
inline void delete_adamw_options (void* x) {
   _delete_adamw_options(x);
  host_exception_handler();
  
}
#endif // RCPP_VERSION
