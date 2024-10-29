library(ignite)
library(torch)
ignite:::rcpp_ignite_run_script_module(torch_randn(1), torch_randn(2))
