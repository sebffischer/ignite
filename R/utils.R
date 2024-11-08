#' @return The object itself, invisibly
#' @keywords internal
assert_script_module <- function(x, arg_name = deparse1(substitute(x))) {
  if (!inherits(x, "script_module")) {
    rlang::abort(sprintf("`%s` must be a traced or scripted module (torch_jit_script_module)", arg_name))
  }
  invisible(x)
}

assert_optim_ignite <- function(x, arg_name = deparse1(substitute(x))) {
  if (!inherits(x, "optim_ignite")) {
    rlang::abort(sprintf("`%s` must be a LibTorch optimizer (optim_ignite)", arg_name))
  }
  invisible(x)
}

priv = function(x) {
  if (!inherits(x, "R6")) {
    return(NULL)
  }

  x[[".__enclos_env__"]][["private"]]
}
