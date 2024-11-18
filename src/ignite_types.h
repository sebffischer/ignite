#pragma once
#include <torch.h>
#include <utility>

namespace ignite {

class optim_sgd {
public:
  std::shared_ptr<void> ptr;
  optim_sgd (void* x);
  optim_sgd (std::shared_ptr<void> x) : ptr(x) {}
  optim_sgd (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim {
public:
  std::shared_ptr<void> ptr;
  optim (void* x);
  optim (std::shared_ptr<void> x) : ptr(x) {}
  optim (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_adam {
public:
  std::shared_ptr<void> ptr;
  optim_adam (void* x);
  optim_adam (std::shared_ptr<void> x) : ptr(x) {}
  optim_adam (SEXP x);
  operator SEXP () const;
  void* get ();
};

// TODO: no need to make it shared pointer as we the memory is owned by the optimizer
class optim_param_groups {
public:
  std::shared_ptr<void> ptr;
  optim_param_groups (void* x);
  optim_param_groups (std::shared_ptr<void> x) : ptr(x) {}
  optim_param_groups (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_param_group {
public:
  std::shared_ptr<void> ptr;
  optim_param_group (void* x);
  optim_param_group (std::shared_ptr<void> x) : ptr(x) {}
  optim_param_group (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_adamw {
public:
  std::shared_ptr<void> ptr;
  optim_adamw (void* x);
  optim_adamw (std::shared_ptr<void> x) : ptr(x) {}
  optim_adamw (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_adagrad {
public:
  std::shared_ptr<void> ptr;
  optim_adagrad (void* x);
  optim_adagrad (std::shared_ptr<void> x) : ptr(x) {}
  optim_adagrad (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_rmsprop {
public:
  std::shared_ptr<void> ptr;
  optim_rmsprop (void* x);
  optim_rmsprop (std::shared_ptr<void> x) : ptr(x) {}
  optim_rmsprop (SEXP x);
  operator SEXP () const;
  void* get ();
};

class stack {
public:
  std::shared_ptr<void> ptr;
  // constructor from a void* pointer;
  stack (void* x);
  // constructor from a shared_ptr<void> pointer;
  stack (std::shared_ptr<void> x) : ptr(x) {}
  // constructor from an R object;
  stack (SEXP x);
  // implicit casting operator
  operator SEXP () const;
  // conversion to a void* pointer;
  void* get ();
};


class adamw_param_groups {
public:
  std::shared_ptr<void> ptr;
  adamw_param_groups (void* x);
  adamw_param_groups (std::shared_ptr<void> x) : ptr(x) {}
  adamw_param_groups (SEXP x);
  operator SEXP () const;
  void* get ();
};


class optim_options {
public:
  void* ptr;
  optim_options (void* x) : ptr(x) {};
  void* get ();
};

}


// TODO: Remove all the functions that are not needed.
// I think for the intermediate types that are not communicated to R we can remove some things.
