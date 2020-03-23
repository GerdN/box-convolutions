#include <torch/extension.h>

torch::Tensor integral_image(
    torch::Tensor input);

torch::Tensor box_convolution_forward(
    torch::Tensor input_integrated,
    torch::Tensor x_min, torch::Tensor x_max,
    torch::Tensor y_min, torch::Tensor y_max,
    const bool normalize, const bool exact);

std::vector<torch::Tensor> box_convolution_backward(
    torch::Tensor input_integrated,
    torch::Tensor x_min, torch::Tensor x_max,
    torch::Tensor y_min, torch::Tensor y_max,
    torch::Tensor grad_output, torch::Tensor output,
    const float reparametrization_h, const float reparametrization_w,
    const bool normalize, const bool exact,
    const bool input_needs_grad,
    const bool x_min_needs_grad, const bool x_max_needs_grad,
    const bool y_min_needs_grad, const bool y_max_needs_grad);

void clip_parameters(
    torch::Tensor x_min, torch::Tensor x_max,
    torch::Tensor y_min, torch::Tensor y_max,
    const double reparametrization_h, const double reparametrization_w,
    const double max_input_h, const double max_input_w, const bool exact);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image", &integral_image, "Integral image");
    m.def("box_convolution_forward" , &box_convolution_forward , "Box convolution, forward" );
    m.def("box_convolution_backward", &box_convolution_backward, "Box convolution, backward");
    m.def("clip_parameters", &clip_parameters, "Box convolution, clip parameters");
}
