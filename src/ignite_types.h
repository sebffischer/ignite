#pragma once
#include <torch.h>

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

class optim_adam {
public:
  std::shared_ptr<void> ptr;
  optim_adam (void* x);
  optim_adam (std::shared_ptr<void> x) : ptr(x) {}
  optim_adam (SEXP x);
  operator SEXP () const;
  void* get ();
};

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

class sgd_param_groups {
public:

  // IMPORTANT: The order of fields is important and must be identical to the implementation in csrc
  struct sgd_param_group_inner {
    std::vector<void*> params;
    double learning_rate;
    double weight_decay;
    double momentum;
    double dampening;
    bool nesterov;

    sgd_param_group_inner(std::vector<void*> params, double learning_rate, double weight_decay, double momentum, double dampening, bool nesterov) : params(params), learning_rate(learning_rate), weight_decay(weight_decay), momentum(momentum), dampening(dampening), nesterov(nesterov) {}
    // Constructor from Rcpp::List
    sgd_param_group_inner(Rcpp::List list) {
        Rcpp::List params_list = Rcpp::as<Rcpp::List>(list["params"]);

        std::vector<void*> params;

        for (auto x : params_list) {
            // Assuming x is a pointer to a torch::Tensor, convert it back to void*
            // x was created with make_xptr<torch::Tensor>
            // cast to an Rcpp::XPtr<torch::Tensor> and then to a void*
            auto xptr = Rcpp::as<Rcpp::XPtr<torch::Tensor>>(x);

            params.push_back(static_cast<void*>(xptr.get()));
        }

        learning_rate = Rcpp::as<double>(list["lr"]);
        weight_decay = Rcpp::as<double>(list["weight_decay"]);
        momentum = Rcpp::as<double>(list["momentum"]);
        dampening = Rcpp::as<double>(list["dampening"]);
        nesterov = Rcpp::as<bool>(list["nesterov"]);
    }
  };
  std::shared_ptr<void> ptr;
  // constructor from a void* pointer;
  sgd_param_groups (void* x);
  // constructor from a shared_ptr<void> pointer;
  sgd_param_groups (std::shared_ptr<void> x) : ptr(x) {}
  // constructor from an R object;
  sgd_param_groups (SEXP x);
  // implicit casting operator
  operator SEXP () const;
  // conversion to a void* pointer;
  void* get ();
};
class adamw_param_groups {
public:

  // IMPORTANT: The order of fields is important and must be identical to the implementation in csrc
  struct adamw_param_group_inner {
    std::vector<void*> params;
    double learning_rate;
    double weight_decay;
    std::pair<double, double> betas;
    double eps;

    adamw_param_group_inner(std::vector<void*> params, double learning_rate, double weight_decay, std::pair<double, double> betas, double eps) : params(params), learning_rate(learning_rate), weight_decay(weight_decay), betas(betas), eps(eps) {}
    // Constructor from Rcpp::List
    adamw_param_group_inner(Rcpp::List list) {
        Rcpp::List params_list = Rcpp::as<Rcpp::List>(list["params"]);

        std::vector<void*> params;

        for (auto x : params_list) {
            // Assuming x is a pointer to a torch::Tensor, convert it back to void*
            // x was created with make_xptr<torch::Tensor>
            // cast to an Rcpp::XPtr<torch::Tensor> and then to a void*
            auto xptr = Rcpp::as<Rcpp::XPtr<torch::Tensor>>(x);

            params.push_back(static_cast<void*>(xptr.get()));
        }

        learning_rate = Rcpp::as<double>(list["lr"]);
        weight_decay = Rcpp::as<double>(list["weight_decay"]);
        Rcpp::NumericVector betas_vec = Rcpp::as<Rcpp::NumericVector>(list["betas"]);
        betas = std::make_pair(betas_vec[0], betas_vec[1]);
        eps = Rcpp::as<double>(list["eps"]);
    }
  };
  std::shared_ptr<void> ptr;
  // constructor from a void* pointer;
  adamw_param_groups (void* x);
  // constructor from a shared_ptr<void> pointer;
  adamw_param_groups (std::shared_ptr<void> x) : ptr(x) {}
  // constructor from an R object;
  adamw_param_groups (SEXP x);
  // implicit casting operator
  operator SEXP () const;
  // conversion to a void* pointer;
  void* get ();
};
}

