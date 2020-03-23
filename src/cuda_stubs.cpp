#include <torch/extension.h>

#include "box_convolution.h" // for `enum class Parameter`

#define STUB_ERROR TORCH_CHECK(false, "box_convolution was compiled withoud CUDA support because " \
                                      "torch.cuda.is_available() was False when you ran setup.py.")

namespace gpu {

void integral_image(torch::Tensor & input, torch::Tensor & output)
{ STUB_ERROR; }

void splitParameters(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac)
{ STUB_ERROR; }

void splitParametersUpdateGradInput(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac)
{ STUB_ERROR; }

void splitParametersAccGradParameters(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac)
{ STUB_ERROR; }

template <bool normalize, bool exact>
void boxConvUpdateOutput(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & area, torch::Tensor & input_integrated, torch::Tensor & output)
{ STUB_ERROR; }

// explicitly instantiate
template void boxConvUpdateOutput<true, true>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template void boxConvUpdateOutput<false, true>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template void boxConvUpdateOutput<true, false>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template void boxConvUpdateOutput<false, false>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template <bool normalize, bool exact>
void boxConvUpdateGradInput(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & area, torch::Tensor & grad_output_integrated, torch::Tensor & tmpArray)
{ STUB_ERROR; }

// explicitly instantiate
template void boxConvUpdateGradInput<true, true>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template void boxConvUpdateGradInput<false, true>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template void boxConvUpdateGradInput<true, false>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template void boxConvUpdateGradInput<false, false>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &);

template <bool exact>
void boxConvAccGradParameters(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & input_integrated, torch::Tensor & tmpArray, Parameter parameter)
{ STUB_ERROR; }

// explicitly instantiate
template void boxConvAccGradParameters<true>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, Parameter);

template void boxConvAccGradParameters<false>(
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &, torch::Tensor &, Parameter);

void clipParameters(
    torch::Tensor & paramMin, torch::Tensor & paramMax,
    const double reparametrization, const double minSize, const double maxSize)
{ STUB_ERROR; }

torch::Tensor computeArea(
    torch::Tensor x_min, torch::Tensor x_max, torch::Tensor y_min, torch::Tensor y_max,
    const bool exact, const bool needXDeriv, const bool needYDeriv)
{ STUB_ERROR; }

}