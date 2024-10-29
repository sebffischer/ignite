#include <Rcpp.h>
#define IGNITE_HEADERS_ONLY
#include <ignite/ignite.h>
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>
#include "exports.h"


void host_exception_handler ()
{
  if (ignite_last_error())
  {
    auto msg = Rcpp::as<std::string>(torch::string(ignite_last_error()));
    ignite_last_error_clear();
    Rcpp::stop(msg);
  }
}

// [[Rcpp::export]]
void ignite_raise_exception ()
{
  raise_exception();
}
