#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"


Rcpp::XPtr<torch::Tensor> get_tensor_ptr(void* tensor) {
    auto xptr = make_xptr<torch::Tensor>(tensor);
    xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
    return xptr;
}

namespace ignite {

void* adamw_param_groups::get() {
  return ptr.get();
}

adamw_param_groups::operator SEXP () const {
  // cast the pointer to a std::vector<void*>
  std::cout << "before" << std::endl;
  // TODO(IMPORTANT): This is actually a struct with a single field which is this vector.
  // C++ does not guarantee the same layout of a type and a struct with one field to this type
  auto v = static_cast<std::vector<void*>*>(ptr.get());

  std::cout << "after" << std::endl;

  // iterate over the pointers and call into rcpp_adamw_param_group for each
  Rcpp::List lst = Rcpp::List::create();
  for (auto* ptr : *v) {
    Rcpp::List lst_inner = Rcpp::List::create();
    lst_inner["params"] = rcpp_ignite_optim_get_param_group_params(&ptr);
    lst_inner["lr"] = rcpp_ignite_optim_get_param_group_lr(&ptr);


    lst.push_back(lst_inner);
  }

  std::cout << "done (?)" << std::endl;

  return lst;
}

adamw_param_groups::adamw_param_groups(SEXP x) {
}

adamw_param_groups::adamw_param_groups (void* x) : ptr(x, rcpp_delete_adamw_param_groups) {};

void* adamw_state::get() {
  return ptr.get();
}

adamw_state::operator SEXP () const {
  return R_NilValue;
}

adamw_state::adamw_state(SEXP x) {
}

adamw_state::adamw_state (void* x) : ptr(x, [](void*) {}) {};


void* adamw_states::get() {
  return ptr.get();
}

adamw_states::operator SEXP () const {

  std::cout << "adamw_states::operator SEXP" << std::endl;

  auto* raw_ptr = ptr.get();

  // this pointer is a std::vector<torch::optim::AdamWParamState*>

  // I want to cast it to a std::vector<void*>
  auto* void_ptr = static_cast<std::vector<void*>*>(raw_ptr);

  // call length of the vector
  std::cout << "length of the vector: " << void_ptr->size() << std::endl;

  // vector is cast properly

  // walk over the pointers and call adamw_state_exp_avg on each
  Rcpp::List lst = Rcpp::List::create();
  for (auto* state : *void_ptr) {
    Rcpp::List lst_inner = Rcpp::List::create();
    lst_inner.push_back(rcpp_adamw_state_exp_avg(&state));
    lst_inner.push_back(rcpp_adamw_state_exp_avg_sq(&state));
    lst_inner.push_back(rcpp_adamw_state_max_exp_avg_sq(&state));
    lst_inner.push_back(rcpp_adamw_state_step(&state));

    lst_inner.names() = Rcpp::CharacterVector::create("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step");
    lst.push_back(lst_inner);
  }
  return lst;
}

adamw_states::adamw_states(SEXP x) {
}

// TODO: Use the correct deleter
adamw_states::adamw_states (void* x) : ptr(x, rcpp_delete_adamw_states) {};


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
optim_param_group::optim_param_group (void* x) : ptr(x, [](void*) {}) {};

void* optim::get() {
  return ptr.get();
}

optim::operator SEXP () const {
  // TODO: don't think we need this
  auto xptr = make_xptr<optim>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite");
  return xptr;
}

optim::optim (SEXP x) : optim{Rcpp::as<Rcpp::XPtr<optim>>(x)->ptr} {}
optim::optim (void* x) : ptr(x, rcpp_delete_optim) {};

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

