#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"


Rcpp::XPtr<torch::Tensor> get_tensor_ptr(void* tensor) {
    auto xptr = make_xptr<torch::Tensor>(tensor);
    xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
    return xptr;
}

namespace ignite {

void* optim_sgd::get() {
  return ptr.get();
}
optim_sgd::operator SEXP () const {
  auto xptr = make_xptr<optim_sgd>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_sgd");
  return xptr;
}
optim_sgd::optim_sgd (SEXP x) : optim_sgd{Rcpp::as<Rcpp::XPtr<optim_sgd>>(x)->ptr} {}
optim_sgd::optim_sgd (void* x) : ptr(x, rcpp_delete_optim_sgd) {};

void* sgd_param_groups::get() {
  return ptr.get();
}


sgd_param_groups::operator SEXP () const {
  // cast the pointer to a std::vector<sgd_param_group_inner>
  auto inner_ptr = std::static_pointer_cast<std::vector<sgd_param_group_inner>>(ptr);
  // iterate over the vector and create a list of lists containing the fields

  Rcpp::List lst = Rcpp::List::create();
  for (const auto& group : *inner_ptr) {
    Rcpp::List param_lst = Rcpp::List::create();
    for (const auto& param : group.params) {
      auto tensor = torch::Tensor(param);
      auto xptr = make_xptr<torch::Tensor>(tensor);
      xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
      param_lst.push_back(xptr);
    }
    Rcpp::List group_lst = Rcpp::List::create(Rcpp::Named("params") = param_lst,
                                              Rcpp::Named("lr") = group.learning_rate,
                                              Rcpp::Named("weight_decay") = group.weight_decay,
                                              Rcpp::Named("momentum") = group.momentum,
                                              Rcpp::Named("dampening") = group.dampening,
                                              Rcpp::Named("nesterov") = group.nesterov);
    lst.push_back(group_lst);
  }
  return lst;
}
// We need this when setting the param groups to e.g. change the learning rate
sgd_param_groups::sgd_param_groups(SEXP x) {
  std::vector<sgd_param_group_inner> param_groups;
  Rcpp::List list(x); // Convert SEXP to Rcpp::List
  for (size_t i = 0; i < list.size(); ++i) {
    Rcpp::List group = Rcpp::as<Rcpp::List>(list[i]); // Access each group
    sgd_param_group_inner inner_group(group); // Use the inner constructor
    param_groups.push_back(inner_group);
  }

ptr = std::make_shared<std::vector<sgd_param_group_inner>>(param_groups);
}
sgd_param_groups::sgd_param_groups (void* x) : ptr(x, rcpp_delete_sgd_param_groups) {};

void* adamw_param_groups::get() {
  return ptr.get();
}

adamw_param_groups::operator SEXP () const {
  auto inner_ptr = std::static_pointer_cast<std::vector<adamw_param_group_inner>>(ptr);

  Rcpp::List lst = Rcpp::List::create();
  for (const auto& group : *inner_ptr) {
    Rcpp::List param_lst = Rcpp::List::create();
    // BUG:
    // HERE IS THE PROBLEM:
    // When we don't convert the param (which is a void*) to a torch::Tensor here,
    // there are no segfaults.

    for (const auto& param : group.params) {
      auto tensor = torch::Tensor(param);
      //auto xptr = make_xptr<torch::Tensor>(tensor);
      // print the address of the tensor
      std::cout << "address of tensor: " << tensor.get() << std::endl;
      //xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
      //param_lst.push_back(xptr);
    }

    // remove all elements from group.params
    //group.params.clear();


    Rcpp::NumericVector betas = Rcpp::NumericVector::create(group.betas.first, group.betas.second);
    Rcpp::List group_lst = Rcpp::List::create(
      Rcpp::Named("params") = param_lst,
                                             Rcpp::Named("lr") = group.learning_rate,
                                             Rcpp::Named("weight_decay") = group.weight_decay,
                                             Rcpp::Named("betas") = betas,
                                             Rcpp::Named("eps") = group.eps,
                                             Rcpp::Named("amsgrad") = group.amsgrad);
    lst.push_back(group_lst);
  }
  return lst;
}

adamw_param_groups::adamw_param_groups(SEXP x) {
  std::vector<adamw_param_group_inner> param_groups;
  Rcpp::List list(x);
  for (size_t i = 0; i < list.size(); ++i) {
    Rcpp::List group = Rcpp::as<Rcpp::List>(list[i]);
    Rcpp::List params_list = Rcpp::as<Rcpp::List>(group["params"]);
    adamw_param_group_inner inner_group(group);
    param_groups.push_back(inner_group);
  }
  ptr = std::make_shared<std::vector<adamw_param_group_inner>>(param_groups);
}

adamw_param_groups::adamw_param_groups (void* x) : ptr(x, rcpp_delete_adamw_param_groups) {};


void* adamw_states::get() {
  return ptr.get();
}

adamw_states::operator SEXP () const {
  auto inner_ptr = std::static_pointer_cast<std::vector<adamw_state_inner>>(ptr);

  Rcpp::List lst = Rcpp::List::create();
  for (const auto& state : *inner_ptr) {
    Rcpp::List group_lst = Rcpp::List::create(
      Rcpp::Named("exp_avg") = get_tensor_ptr(state.exp_avg),
      Rcpp::Named("exp_avg_sq") = get_tensor_ptr(state.exp_avg_sq),
      Rcpp::Named("max_exp_avg_sq") = get_tensor_ptr(state.max_exp_avg_sq),
      Rcpp::Named("step") = state.step);
    lst.push_back(group_lst);
  }
  return lst;
}

adamw_states::adamw_states(SEXP x) {
  std::vector<adamw_state_inner> states;
  Rcpp::List list(x);
  for (size_t i = 0; i < list.size(); ++i) {
    Rcpp::List state = Rcpp::as<Rcpp::List>(list[i]);
    adamw_state_inner inner_state(state);
    states.push_back(inner_state);
  }
  ptr = std::make_shared<std::vector<adamw_state_inner>>(states);
}

adamw_states::adamw_states (void* x) : ptr(x, rcpp_delete_adamw_states) {};

// adam

void* optim_adam::get() {
  return ptr.get();
}
optim_adam::operator SEXP () const {
  auto xptr = make_xptr<optim_adam>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_adam");
  return xptr;
}
optim_adam::optim_adam (SEXP x) : optim_adam{Rcpp::as<Rcpp::XPtr<optim_adam>>(x)->ptr} {}
optim_adam::optim_adam (void* x) : ptr(x, rcpp_delete_optim_adam) {};

void* optim_param_groups::get() {
  return ptr.get();
}
optim_param_groups::operator SEXP () const {
  auto xptr = make_xptr<optim_param_groups>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_param_groups");
  return xptr;
}
optim_param_groups::optim_param_groups (SEXP x) : optim_param_groups{Rcpp::as<Rcpp::XPtr<optim_param_groups>>(x)->ptr} {}
optim_param_groups::optim_param_groups (void* x) : ptr(x, rcpp_delete_optim_param_groups) {};


void* optim_param_group::get() {
  return ptr.get();
}
optim_param_group::operator SEXP () const {
  auto xptr = make_xptr<optim_param_group>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_param_group");
  return xptr;
}
optim_param_group::optim_param_group (SEXP x) : optim_param_group{Rcpp::as<Rcpp::XPtr<optim_param_group>>(x)->ptr} {}
optim_param_group::optim_param_group (void* x) : ptr(x, rcpp_delete_optim_param_group) {};



void* optim_adagrad::get() {
  return ptr.get();
}
optim_adagrad::operator SEXP () const {
  auto xptr = make_xptr<optim_adagrad>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_adagrad");
  return xptr;
}
optim_adagrad::optim_adagrad (SEXP x) : optim_adagrad{Rcpp::as<Rcpp::XPtr<optim_adagrad>>(x)->ptr} {}
optim_adagrad::optim_adagrad (void* x) : ptr(x, rcpp_delete_optim_adagrad) {};

void* optim_rmsprop::get() {
  return ptr.get();
}
optim_rmsprop::operator SEXP () const {
  auto xptr = make_xptr<optim_rmsprop>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_rmsprop");
  return xptr;
}
optim_rmsprop::optim_rmsprop (SEXP x) : optim_rmsprop{Rcpp::as<Rcpp::XPtr<optim_rmsprop>>(x)->ptr} {}
optim_rmsprop::optim_rmsprop (void* x) : ptr(x, rcpp_delete_optim_rmsprop) {};

void* optim_adamw::get() {
  return ptr.get();
}
optim_adamw::operator SEXP () const {
  auto xptr = make_xptr<optim_adamw>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_adamw");
  return xptr;
}
optim_adamw::optim_adamw (SEXP x) : optim_adamw{Rcpp::as<Rcpp::XPtr<optim_adamw>>(x)->ptr} {}
optim_adamw::optim_adamw (void* x) : ptr(x, rcpp_delete_optim_adamw) {};
}

