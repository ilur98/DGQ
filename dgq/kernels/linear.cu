#include "include/linear.h"
#include "include/common.h"
#include "include/cutlass_extensions/epilogue/threadblock/epilogue_per_row_per_col_scale.h"
#include "include/cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

__global__ void dequantize_u4_to_s8x4(int8_t const* source, int8_t* result, int8_t* scales, int8_t* zeros, int n, int groupsize) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >=n) return;
  int n2 = tid / groupsize;
  int8_t zero = reinterpret_cast<int8_t const*> (zeros)[n2];
  int8_t scale = reinterpret_cast<int8_t const*> (scales)[n2];
  for(int i = 0; i < 4; i++){
    int tid_d = tid * 8 + 2 * i;
    int tid_s = tid * 4 + i;

    int8_t s = source[tid_s];
    
    result[tid_d] = (((s >> 4) & 0xf) - zero) * scale;
    result[tid_d+1] = ((s & 0xf) - zero) * scale;
  }

  __syncthreads();
}

#define THREAD_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

  return optimal_block_num;
}

void launch_dequant(torch::Tensor src, torch::Tensor & tgt,  int8_t* scales, int8_t* zeros, int groupsize) {
  int n = (at::numel(src) + 4 - 1) / 4;
  dequantize_u4_to_s8x4<<<GET_BLOCKS(n), THREAD_PER_BLOCK, 0, c10::cuda::getCurrentCUDAStream()>>>(src.data_ptr<int8_t>(), tgt.data_ptr<int8_t>(), scales, zeros, n, groupsize);
}

// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w4_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP32
                                      torch::Tensor alpha,  // FP32
                                      torch::Tensor beta,   // FP32
                                      torch::Tensor scales8,
                                      torch::Tensor zeros,
                                      int cin, 
                                      int cout,
                                      int groupsize
) {
  auto M = input.size(0);
  auto N = (int64_t) cout;
  auto K = (int64_t) cin;

  auto weight_int8 = torch::empty({N, K},
                        torch::dtype(torch::kChar).device(weight.device()));
  auto source = weight.data_ptr<int8_t>();
  auto result = weight_int8.data_ptr<int8_t>();
  auto scales_ptr = scales8.data_ptr<int8_t>();
  auto zeros_ptr  = zeros.data_ptr<int8_t>();

  launch_dequant(weight, weight_int8, scales_ptr, zeros_ptr, groupsize);
  using ElementOutput = float;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  auto device = input.device();
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using OperatorClass   = cutlass::arch::OpClassTensorOp;
  using DefaultGemmConf = typename cutlass::gemm::device::
      DefaultGemmConfiguration<OperatorClass, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput, ElementComputeEpilogue>;
  using InstructionShape = typename DefaultGemmConf::InstructionShape;
  using GemmOp           = typename DefaultGemmConf::Operator;
  using EpilogueOp       = typename DefaultGemmConf::EpilogueOutputOp;
  // using ActivationOp     = typename ;
  // only TN is supported (s8 * s8 + s32)
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInputA,
                                      cutlass::layout::RowMajor,
                                      DefaultGemmConf::kAlignmentA,
                                      ElementInputB,
                                      cutlass::layout::ColumnMajor,
                                      DefaultGemmConf::kAlignmentB,
                                      ElementOutput,
                                      cutlass::layout::RowMajor,
                                      ElementAccumulator,
                                      OperatorClass,
                                      cutlass::arch::Sm80,
                                      cutlass::gemm::GemmShape<256, 128, 64>,
                                      cutlass::gemm::GemmShape<64, 64, 64>, 
                                      cutlass::gemm::GemmShape<16, 8, 32>,
                                      EpilogueOp,
                                      ThreadblockSwizzle,
                                      3,
                                      true,
                                      GemmOp>::GemmKernel;
  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
        typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
        typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
        GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
        GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
        cutlass::sizeof_bits<ElementComputeEpilogue>::value>,
        ElementComputeEpilogue>;

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      cutlass::gemm::GemmShape<256, 128, 64>,
      GemmKernel_::kThreadCount,
      AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator,
      ElementAccumulator,
      ElementComputeEpilogue,
      EpilogueOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

  typename EpilogueOp::Params linear_scaling_params;  // TODO(mseznec): right now it's unused (scaling is done in
                                                        // visitor, no activation needed)
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});
  auto quant_mode = cutlass::epilogue::QuantMode::PerChannelQuant;
  // auto input_ref = (const ElementInputA*)input.data_ptr<ElementInputA>();
  // auto weight_ref = (const ElementInputB*)weight_int8.data_ptr<ElementInputB>();
  auto alpha_ref = (const ElementComputeEpilogue*)alpha.data_ptr();
  auto beta_ref = (const ElementComputeEpilogue*)beta.data_ptr();
  // auto out_ref = out.data_ptr<ElementOutput>();
  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);
  // auto alpha_size = cutlass::MatrixCoord(N);
  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight_int8.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      (ElementOutput*)out.data_ptr(), LayoutOutput::packed(output_size));
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K},
                                1,
                                input_ref,
                                weight_ref,
                                quant_mode,
                                {reinterpret_cast<ElementComputeEpilogue*>(const_cast<ElementComputeEpilogue*>(alpha_ref)), 0},
                                // {reinterpret_cast<ElementComputeEpilogue*>(const_cast<ElementComputeEpilogue*>(beta_ref)), 0},
                                {nullptr, 0},
                                out_ref,
                                out_ref,
                                0,
                                0,
                                typename EpilogueVisitor::Arguments(linear_scaling_params, 0, 0, 0)};
  Gemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (gemm_op.get_workspace_size(arguments) > 0) {
        arguments.batch_count = 1;
    }
  
  auto can_implement = gemm_op.can_implement(arguments);
  if (can_implement != cutlass::Status::kSuccess) {
      std::string err_msg = "int8gemm cutlass kernel will fail for params. Error: ";
      throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
  }

  auto init_status = gemm_op.initialize(arguments, 0, stream);
  if (init_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to initialize cutlass int8 gemm. Error: ";
      throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
  }
  auto run_status = gemm_op.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to run cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(run_status));
      throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
  }
  return out;
}

// used by q_proj, k_proj, v_proj, return INT8
torch::Tensor linear_a8_w4_b8_o8(torch::Tensor input,  // INT8
                                 torch::Tensor weight, // INT8
                                 torch::Tensor bias,   // FP32
                                  torch::Tensor alpha,  // FP32
                                  torch::Tensor beta,   // FP32
                                  torch::Tensor scales8,
                                  torch::Tensor zeros,
                                  int cin, 
                                  int cout,
                                  int groupsize
) {
  auto M = input.size(0);
  auto N = (int64_t) cout;
  auto K = (int64_t) cin;

  auto weight_int8 = torch::empty({N, K},
                        torch::dtype(torch::kChar).device(weight.device()));
  auto source = weight.data_ptr<int8_t>();
  auto result = weight_int8.data_ptr<int8_t>();
  auto scales_ptr = scales8.data_ptr<int8_t>();
  auto zeros_ptr  = zeros.data_ptr<int8_t>();

  launch_dequant(weight, weight_int8, scales_ptr, zeros_ptr, groupsize);
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  auto device = input.device();
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using OperatorClass   = cutlass::arch::OpClassTensorOp;
  using DefaultGemmConf = typename cutlass::gemm::device::
      DefaultGemmConfiguration<OperatorClass, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput, ElementComputeEpilogue>;
  using ThreadblockShape = typename DefaultGemmConf::ThreadblockShape;
  using InstructionShape = typename DefaultGemmConf::InstructionShape;
  using WarpShape        = typename DefaultGemmConf::WarpShape;
  using GemmOp           = typename DefaultGemmConf::Operator;
  // using EpilogueOp       = cutlass::epilogue::thread::LinearCombination<ElementOutput, 256 / cutlass::sizeof_bits<ElementComputeEpilogue>::value, ElementAccumulator,
      // ElementComputeEpilogue>;
  using EpilogueOp       = typename DefaultGemmConf::EpilogueOutputOp;
  // using ActivationOp     = typename ;
  // only TN is supported (s8 * s8 + s32)
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInputA,
                                      cutlass::layout::RowMajor,
                                      DefaultGemmConf::kAlignmentA,
                                      ElementInputB,
                                      cutlass::layout::ColumnMajor,
                                      DefaultGemmConf::kAlignmentB,
                                      ElementOutput,
                                      cutlass::layout::RowMajor,
                                      ElementAccumulator,
                                      OperatorClass,
                                      cutlass::arch::Sm80,
                                      cutlass::gemm::GemmShape<256, 128, 64>,
                                      cutlass::gemm::GemmShape<64, 64, 64>, 
                                      cutlass::gemm::GemmShape<16, 8, 32>,
                                      EpilogueOp,
                                      ThreadblockSwizzle,
                                      3,
                                      true,
                                      GemmOp>::GemmKernel;
  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
        typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
        typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
        GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
        8,
        32>,
        ElementComputeEpilogue>;

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      cutlass::gemm::GemmShape<256, 128, 64>,
      GemmKernel_::kThreadCount,
      AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator,
      ElementAccumulator,
      ElementComputeEpilogue,
      EpilogueOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

  typename EpilogueOp::Params linear_scaling_params;  // TODO(mseznec): right now it's unused (scaling is done in
                                                        // visitor, no activation needed)
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});
  auto quant_mode = cutlass::epilogue::QuantMode::PerChannelQuant;
  auto alpha_ref = (const ElementComputeEpilogue*)alpha.data_ptr();
  auto beta_ref = (const ElementComputeEpilogue*)beta.data_ptr();
  // auto out_ref = out.data_ptr<ElementOutput>();
  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);
  // auto alpha_size = cutlass::MatrixCoord(N);
  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight_int8.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      (ElementOutput*)out.data_ptr(), LayoutOutput::packed(output_size));
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K},
                                1,
                                input_ref,
                                weight_ref,
                                quant_mode,
                                {reinterpret_cast<ElementComputeEpilogue*>(const_cast<ElementComputeEpilogue*>(alpha_ref)), 0},
                                {reinterpret_cast<ElementComputeEpilogue*>(const_cast<ElementComputeEpilogue*>(beta_ref)), 0},
                                out_ref,
                                out_ref,
                                0,
                                0,
                                typename EpilogueVisitor::Arguments(linear_scaling_params, 0, 0, 0)};
  Gemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

//   if (gemm_op.get_workspace_size(arguments) > 0) {
//         arguments.batch_count = 1;
//     }
  
  auto can_implement = gemm_op.can_implement(arguments);
  if (can_implement != cutlass::Status::kSuccess) {
      std::string err_msg = "int8gemm cutlass kernel will fail for params. Error: ";
      throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
  }

  auto init_status = gemm_op.initialize(arguments, 0, stream);
  if (init_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to initialize cutlass int8 gemm. Error: ";
      throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
  }
  auto run_status = gemm_op.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to run cutlass int8 gemm. Error: ";
      throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
  }
  return out;
}