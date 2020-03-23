#include <ciso646> // && -> and, || -> or etc.

enum class Parameter {xMin, xMax, yMin, yMax};

namespace cpu {

void splitParameters(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac);

void splitParametersUpdateGradInput(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac);

void splitParametersAccGradParameters(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac);

template <bool normalize, bool exact>
void boxConvUpdateOutput(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & area, torch::Tensor & input_integrated, torch::Tensor & output);

template <bool normalize, bool exact>
void boxConvUpdateGradInput(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & area, torch::Tensor & grad_output_integrated, torch::Tensor & tmpArray);

template <bool exact>
void boxConvAccGradParameters(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & input_integrated, torch::Tensor & tmpArray, Parameter parameter);

void clipParameters(
    torch::Tensor & paramMin, torch::Tensor & paramMax,
    const double reparametrization, const double minSize, const double maxSize);

torch::Tensor computeArea(
    torch::Tensor x_min, torch::Tensor x_max, torch::Tensor y_min, torch::Tensor y_max,
    const bool exact, const bool needXDeriv = true, const bool needYDeriv = true);

}

namespace gpu {

void splitParameters(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac);

void splitParametersUpdateGradInput(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac);

void splitParametersAccGradParameters(
    torch::Tensor & x_min   , torch::Tensor & x_max   , torch::Tensor & y_min   , torch::Tensor & y_max   ,
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac);

template <bool normalize, bool exact>
void boxConvUpdateOutput(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & area, torch::Tensor & input_integrated, torch::Tensor & output);

template <bool normalize, bool exact>
void boxConvUpdateGradInput(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & area, torch::Tensor & grad_output_integrated, torch::Tensor & tmpArray);

template <bool exact>
void boxConvAccGradParameters(
    torch::Tensor & xMinInt , torch::Tensor & xMaxInt , torch::Tensor & yMinInt , torch::Tensor & yMaxInt ,
    torch::Tensor & xMinFrac, torch::Tensor & xMaxFrac, torch::Tensor & yMinFrac, torch::Tensor & yMaxFrac,
    torch::Tensor & input_integrated, torch::Tensor & tmpArray, Parameter parameter);

void clipParameters(
    torch::Tensor & paramMin, torch::Tensor & paramMax,
    const double reparametrization, const double minSize, const double maxSize);

torch::Tensor computeArea(
    torch::Tensor x_min, torch::Tensor x_max, torch::Tensor y_min, torch::Tensor y_max,
    const bool exact, const bool needXDeriv = true, const bool needYDeriv = true);

}