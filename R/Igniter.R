#' @title Igniter
#' @description
#' An R6 class that performs optimization steps and predictions with a given network, loss function, target and optimizer.
#' @export
Igniter = R6::R6Class("Igniter",
  public = list(
    #' @field network (`torch_script_module`)\cr
    #' The network to use.
    network = NULL,
    #' @field loss_fn (`torch_script_module`)\cr
    #' The loss function to use.
    loss_fn = NULL,
    #' @field optimizer `torch_optimizer`
    #' The optimizer to use.
    optimizer = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param network (`torch_script_module`)\cr
    #'   The jitted network.
    #' @param loss_fn (`torch_script_module`)\cr
    #'   The jitted loss function.
    #''
    initialize = function(network, loss_fn, optimizer) {
      assert_script_module(network)
      assert_script_module(loss_fn)
      assert_optim_ignite(optimizer)
      self$optimizer = optimizer$ptr
      self$network = priv(attr(network, "module"))$ptr
      self$loss_fn = priv(attr(loss_fn, "module"))$ptr
    },

    #' @description Set the network to training mode.
    #' @return The object itself, invisibly
    train = function() {
      self$network$train()
      invisible(self)
    },
    #' @description Set the network to evaluation mode
    #' @return The object itself, invisibly
    eval = function() {
      self$network$eval()
      invisible(self)
    },

    # TODO: Should maybe not return tensors but R vectors,
    #' maybe this also also make this configurable.

    #' Perform an optimization step
    #'
    #' Performs a forward pass through the network, calculates the loss,
    #' backpropagates gradients and updates model parameters using the optimizer.
    #'
    #' Note that you need to ensure that the network is in `"train"`-mode beforehand.
    #'
    #' @param  input `list(torch_tensor)`
    #'  Inputs to the network.
    #' @param target `torch_tensor`
    #'  Targets for the loss function.
    #'
    #' @return A `list()` containing the loss value and network output as `torch_tensor` objects
    opt_step = function(input, target) {
      rcpp_ignite_opt_step(self$network, self$loss_fn, input, target, self$optimizer)
    },
    #' Perform a prediction step
    #'
    #' Performs a forward pass through the network, but does not calculate the loss.
    #'
    #' Note that you need to ensure that the network is in `"eval"`-mode beforehand.
    #'
    #' @return A `torch_tensor` containing the prediction.
    predict_step = function() {
      rcpp_ignite_predict_step(self$network, self$input)
    }
  )
)
