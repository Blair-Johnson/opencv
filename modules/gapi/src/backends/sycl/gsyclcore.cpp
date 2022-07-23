#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/sycl/core.hpp> //see what this points to
#include "backends/sycl/gsyclcore.hpp"

// Define GAPI SYCL Kernels here

//GAPI_SYCL_KERNEL(GSYCLAdd, cv::gapi::core::GAdd)
//{
//  static void run()
//  {
//  }
//};


// Define GAPI Kernel package here

cv::gapi::GKernelPackage cv::gapi::core::sycl::kernels()
{
  static auto pkg = cv::gapi::kernels
    < //GSYCLAdd
    >();
  return pkg;
}
