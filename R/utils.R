#' @return The object itself, invisibly
#' @keywords internal
assert_script_module <- function(x, arg_name = deparse1(substitute(x))) {
  if (!inherits(x, "script_module")) {
    rlang::abort(sprintf("`%s` must be a traced or scripted module (torch_jit_script_module)", arg_name))
  }
  invisible(x)
}
