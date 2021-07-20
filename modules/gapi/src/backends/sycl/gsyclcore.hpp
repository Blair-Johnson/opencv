#ifndef OPENCV_GAPI_GSYCLCORE_HPP
#define OPENCV_GAPI_GSYCLCORE_HPP

#include <map>
#include <string>

#include <opencv2/gapi/sycl/gsyclkernel.hpp>

namespace cv { namespace gimpl {

  void loadGSYCLCore(std::map<std::string, cv::GSYCLKernel> &kmap);
}
}

#endif // OPENCV_GAPI_GSYCLCORE_HPP
