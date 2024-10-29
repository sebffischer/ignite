#' @title Igniter
#'
#' @description
#' An R6 class that performs optimization steps and predictions with a given network, loss function, target and optimizer.
#'
#' **Important**: No input checks are performed in the prediction and optimization step.
#' @export
Igniter = R6::R6Class(
  "Igniter",
  public = list(
    network = NULL,
    loss_fn = NULL,
    input = NULL,
    target = NULL,
    optimizer = NULL,
    initialize = function(network, loss_fn, target, optimizer) {
      self$network = assert_script_module(network)
      self$loss_fn = assert_script_module(loss_fn)
      self$target <- target
      self$optimizer <- optimizer
    },

    #' Set the network to training mode
    #' Sets the network to training mode, e.g. dropout is active etc.
    #' @return The object itself, invisibly
    train = function() {
      self$network$train()
      invisible(self)
    },
    #' Set the network to evaluation mode
    #' Sets the network to evaluation mode, e.g. dropout is inactive etc.
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
    #' @typed  input: `list(torch_tensor)`
    #'  Inputs to the network.
    #'
    #' @return A `list()` containing the loss value and network output as `torch_tensor` objects
    opt_step = function(input) {
      rcpp_ignite_opt_step(self$network, self$loss_fn, input, self$target, self$optimizer)
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
