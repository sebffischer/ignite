#' @title Abstract Base Class for LibTorch Optimizers
#' @description
#' Abstract base class for creating optimizers implemented in C++.
#' It is assumed that `self$ptr` is a pointer to the optimizer.
#' Failing to implement this contract will lead to undefined behavior and possibly segfaults when
#' this expected, e.g. by the [`Igniter`] class.
#' @inheritParams torch::optimizer
#' @section State Dict:
#' The `$state_dict()` method returns a list with two elements:
#' - `param_groups`: A list of parameter groups.
#'   Each parameter group contains a field:
#'   - `params`: An integer vector indicating the indices of the parameters in the optimizer.
#'   - other arbitrary fields such as `lr`, `weight_decay`, etc.
#' - `states`: A list of optimizer states. The length of this list is the same as the number of parameters.
#'    The structure of the optimizer states is specific to the optimizer.
#' @section Loading State Dict:
#' The `$load_state_dict()` method loads the state dict.
#' @export
optimizer_ignite = function (name = NULL, ..., private = NULL,
  active = NULL, parent_env = parent.frame()) {
  get_ptr = list(...)$get_ptr
  if (!is.function(get_ptr)) {
    stop()
  }
  # TODO: I think we should probably not call this function as it initializes some things
  # we don't need

  torch::optimizer(
    name = c(name, "optim_ignite"),
    inherit = NULL,
    ...,
    private = private,
    active = active,
    parent_env = parent_env
  )
}

#' @title SGD Optimizer as implemented in LibTorch
#' @inheritParams torch::optim_sgd
#' @export
optim_ignite_sgd <- optimizer_ignite(
  "optim_ignite_sgd",
  initialize = function(params, lr, momentum = 0, dampening = 0, weight_decay = 0,
      nesterov = FALSE) {
    private$.ptr <- rcpp_ignite_sgd(params, lr, momentum, dampening, weight_decay, nesterov)
  },
  get_ptr = function() {
    # TODO: make this private
    self$ptr
  },
  step = function() {
    rcpp_ignite_sgd_step(self$ptr)
  },
  zero_grad = function() {
    rcpp_ignite_sgd_zero_grad(self$ptr)
  },
  state_dict = function() {
    # TODO:

  },
  load_state_dict = function(state_dict) {
    # TODO:
  },
  active = list(
    ptr = function() {
      private$.ptr
    },
    param_groups = function(rhs) {
      if (!missing(rhs)) {
        rcpp_ignite_sgd_set_param_groups(self$get_ptr(), rhs)
      }
      rcpp_ignite_sgd_get_param_groups(self$get_ptr())
    }
  )
)

#' @title SGD Optimizer as implemented in LibTorch
#' @inheritParams torch::optim_adam
#' @export
optim_ignite_adam <- optimizer_ignite(
  "optim_ignite_adam",
  initialize = function(params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
    weight_decay = 0, amsgrad = FALSE) {

    self$ptr <- rcpp_ignite_adam(params, list(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad))
  },
  get_ptr = function() {
    self$ptr
  },
  step = function() {
    rcpp_ignite_adam_step(self$ptr)
  },
  zero_grad = function() {
    rcpp_ignite_adam_zero_grad(self$ptr)
  }
)

#' @export
#' @title SGD Optimizer as implemented in LibTorch
#' @inheritParams torch::optim_adam
#' @export
optim_ignite_adamw <- optimizer_ignite(
  "optim_ignite_adamw",
  initialize = function(params, lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8,
                       weight_decay = 1e-2, amsgrad = FALSE) {
    self$ptr <- rcpp_ignite_adamw(params = params, lr = lr, beta1 = betas[1], beta2 = betas[2], eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
  },
  get_ptr = function() {
    self$ptr
  },
  state_dict = function() {
    # the param_groups actually contain the parameters that are optimized.
    # But we don't want to return them as part of the state dict.
    # Therefore, we unlist all the parameters and store the indices in the state dict.
    param_groups = self$param_groups
    parameters <- unlist(lapply(param_groups, function(x) x$params))
    parameters <- sapply(parameters, torch:::xptr_address)


    param_groups = lapply(param_groups, function(group) {
      group_param <- sapply(group$params, torch:::xptr_address)
      group$params <- match(group_param, parameters)
      group
    })

    list(
      param_groups = param_groups,
      states = rcpp_ignite_adamw_get_states(self$ptr)
    )
  },
  load_state_dict = function(state_dict) {
    #if (!length(self$param_groups) == length(state_dict$param_groups)) {
    #  torch:::value_error("Loaded state dict has a different number of parameter groups")
    #}
    #for (i in seq_along(self$param_groups)) {
    #  if (!length(self$param_groups[[i]]$params) == length(state_dict$param_groups[[i]]$params)) {
    #    torch:::value_error("Loaded state dict has contains a parameter group that doesn't match the size of optimizers group.")
    #  }
    #}
    rcpp_ignite_set_adamw_states(self$ptr, state_dict)
  },
  step = function() {
    rcpp_ignite_adamw_step(self$ptr)
  },
  zero_grad = function() {
    rcpp_ignite_adamw_zero_grad(self$ptr)
  },
  active = list(
    # TODO: Add the params as an integer vector.
    param_groups = function(rhs) {
      if (!missing(rhs)) {
        rcpp_ignite_sgd_set_param_groups(self$get_ptr(), rhs)
      }
      rcpp_ignite_adamw_get_param_groups(self$get_ptr())
    }
  )
)

#' @export
optim_ignite_adagrad <- optimizer_ignite(
  "optim_ignite_adagrad",
  initialize = function(params, lr = 1e-2, lr_decay = 0, weight_decay = 0,
                       initial_accumulator_value = 0, eps = 1e-10) {
    self$ptr <- rcpp_ignite_adagrad(params, lr, lr_decay, weight_decay,
                                   initial_accumulator_value, eps)
  },
  get_ptr = function() {
    self$ptr
  },
  step = function() {
    rcpp_ignite_adagrad_step(self$ptr)
  },
  zero_grad = function() {
    rcpp_ignite_adagrad_zero_grad(self$ptr)
  }
)

#' @export
optim_ignite_rmsprop <- optimizer_ignite(
  "optim_ignite_rmsprop",
  initialize = function(params, lr = 1e-2, alpha = 0.99, eps = 1e-8, weight_decay = 0,
                       momentum = 0, centered = FALSE) {
    self$ptr <- rcpp_ignite_rmsprop(params, lr, alpha, eps, weight_decay, momentum, centered)
  },
  get_ptr = function() {
    self$ptr
  },
  step = function() {
    rcpp_ignite_rmsprop_step(self$ptr)
  },
  zero_grad = function() {
    rcpp_ignite_rmsprop_zero_grad(self$ptr)
  }
)
