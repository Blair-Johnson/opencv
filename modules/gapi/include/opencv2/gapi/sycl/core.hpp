#ifndef OPENCV_GAPI_SYCL_CORE_API_HPP
#define OPENCV_GAPI_SYCL_CORE_API_HPP

#include <opencv2/core/cvdef.h> // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace core {
namespace sycl {

GAPI_EXPORTS_W cv::gapi::GKernelPackage kernels();

} // namespace sycl
} // namespace core
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_SYCL_CORE_API_HPP
