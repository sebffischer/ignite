#' @export
optim_ignite_sgd <- torch::optimizer(
  "optim_ignite_sgd",
  initialize = function(params, lr, momentum = 0, dampening = 0, weight_decay = 0,
                        nesterov = FALSE) {
    self$ptr <- rcpp_ignite_sgd(params, lr, momentum, dampening, weight_decay,
                              nesterov)
  },
  step = function() {
    optim_sgd_step(self$ptr)
  },
  zero_grad = function() {
    optim_sgd_zero_grad(self$ptr)
  }
)

#' @export
optim_ignite_adam <- torch::optimizer(
  "optim_ignite_adam",
  initialize = function(params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
                       weight_decay = 0, amsgrad = FALSE) {
    self$ptr <- rcpp_ignite_adam(params, lr, beta1, beta2, eps, weight_decay, amsgrad)
  },
  step = function() {
    optim_adam_step(self$ptr)
  },
  zero_grad = function() {
    optim_adam_zero_grad(self$ptr)
  }
)

#' @export
optim_ignite_adamw <- torch::optimizer(
  "optim_ignite_adamw",
  initialize = function(params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
                       weight_decay = 0.01, amsgrad = FALSE) {
    self$ptr <- rcpp_ignite_adamw(params, lr, beta1, beta2, eps, weight_decay, amsgrad)
  },
  step = function() {
    optim_adamw_step(self$ptr)
  },
  zero_grad = function() {
    optim_adamw_zero_grad(self$ptr)
  }
)

#' @export
optim_ignite_adagrad <- torch::optimizer(
  "optim_ignite_adagrad",
  initialize = function(params, lr, lr_decay = 0, weight_decay = 0,
                       initial_accumulator_value = 0, eps = 1e-10) {
    self$ptr <- rcpp_ignite_adagrad(params, lr, lr_decay, weight_decay,
                                   initial_accumulator_value, eps)
  },
  step = function() {
    optim_adagrad_step(self$ptr)
  },
  zero_grad = function() {
    optim_adagrad_zero_grad(self$ptr)
  }
)

#' @export
optim_ignite_rmsprop <- torch::optimizer(
  "optim_ignite_rmsprop",
  initialize = function(params, lr, alpha = 0.99, eps = 1e-8, weight_decay = 0,
                       momentum = 0, centered = FALSE) {
    self$ptr <- rcpp_ignite_rmsprop(params, lr, alpha, eps, weight_decay,
                                   momentum, centered)
  },
  step = function() {
    optim_rmsprop_step(self$ptr)
  },
  zero_grad = function() {
    optim_rmsprop_zero_grad(self$ptr)
  }
)
