#ifndef LINEAR_H
#define LINEAR_H
#include <torch/types.h>

// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w4_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP32
                                       torch::Tensor alpha,  // BF16
                                       torch::Tensor beta,   // BF16
                                       torch::Tensor scales8,
                                       torch::Tensor zeros,
                                       int cin,
                                       int cout,
                                       int groupsize
);

// used by q_proj, k_proj, v_proj, return INT8
torch::Tensor linear_a8_w4_b8_o8(torch::Tensor input,  // INT8
                                torch::Tensor weight, // INT8
                                torch::Tensor bias,   // INT8
                                torch::Tensor alpha,  // BF16
                                torch::Tensor beta,   // BF16
                                torch::Tensor scales8,
                                torch::Tensor zeros,
                                int cin,
                                int cout,
                                int groupsize
);

#endif // LINEAR_HS