#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"

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
}

