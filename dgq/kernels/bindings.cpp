#include "include/bmm.h"
#include "include/linear.h"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_a8_w4_bfp32_ofp32", &linear_a8_w4_bfp32_ofp32,
        "Linear (I8-OFP32)");
  m.def("linear_a8_w4_b8_o8", &linear_a8_w4_b8_o8, "Linear (INT8)");
  m.def("bmm_s8t_s8n_f32t", &bmm_s8t_s8n_f32t, "BMM (INT8 I FP32 O) A x B.T");
}
