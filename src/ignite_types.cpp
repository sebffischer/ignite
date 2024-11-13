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

  // TODa(DANIEL): We need to take care of deleting these options at the end
  auto v = reinterpret_cast<std::vector<void*>*>(ptr.get());

  std::cout << "after" << std::endl;

  // iterate over the pointers and call into rcpp_adamw_param_group for each
  Rcpp::List lst = Rcpp::List::create();
  // TODO(DANIEL): This will not work on windows, but we can keep it for now.
  for (auto* ptr : *v) {
    Rcpp::List lst_inner = Rcpp::List::create();
    lst_inner["params"] = rcpp_ignite_optim_get_param_group_params(&ptr);

    auto options = rcpp_ignite_adamw_get_param_group_options(&ptr);

    // now cast the optinos to adamw_options::adamw_options_inner
    // static cast is ok here as we know the type

    // TODO(IMPORTANT): We are now the owner of the heap allocated options and need to make sure to
    // delete it.
    // otherwise we have a memory leak
    auto options_inner = *reinterpret_cast<adamw_options::adamw_options_inner*>(options.get());

    // TODO: Maybe put this into the adam_options -> SEXP conversion
    lst_inner["lr"] = options_inner.lr;
    lst_inner["weight_decay"] = options_inner.weight_decay;
    // make a vector from the tuple
    lst_inner["betas"] = Rcpp::NumericVector::create(std::get<0>(options_inner.betas), std::get<1>(options_inner.betas));
    lst_inner["eps"] = options_inner.eps;
    lst_inner["amsgrad"] = options_inner.amsgrad;

    delete options_inner;

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

// TODO: Is this correct? the memory of the state is managed by the optimizer (I think)
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
    Rcpp::List lst_inner = Rcpp::as<Rcpp::List>(rcpp_ignite_adamw_get_state(&state));
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
// because this is only a pointer I think we don't need to delete it.

// TODO: Is this correct? the memory of the param_group is managed by the optimizer (I think)
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

void* optim_options::get() {
  return ptr;
}


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

void* adamw_options::get() {
  // cast to void*
  return static_cast<void*>(ptr.get());
}
adamw_options::operator SEXP () const {
  return R_NilValue;
  //Rcpp::List lst = Rcpp::List::create();
  //auto x = get();
  //lst["lr"] = x->lr;
  //lst["weight_decay"] = x->weight_decay;
  //lst["betas"] = x->betas;
  //lst["eps"] = x->eps;
  //lst["amsgrad"] = x->amsgrad;
  //return lst;
}
adamw_options::adamw_options (void* x) : ptr(x, rcpp_delete_adamw_options) {};
adamw_options::adamw_options (SEXP x) {
  // TODO: We need to implement this for setting the param group options
}

}

