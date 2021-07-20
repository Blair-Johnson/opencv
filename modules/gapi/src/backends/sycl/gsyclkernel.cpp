#include <cassert>
#include <opencv2/core/ocl.hpp>
#include <opencv2/gapi/sycl/gsyclkernel.hpp>

cv::GSYCLContext::GSYCLContext(sycl::queue& sycl_queue)
    m_queue(sycl_queue)
{
    m_context = m_queue.get_context();
}

sycl::queue& cv::GSYCLContext::getQueue()
{
    return m_queue;
}

// FIXME: Unsure if additional modifications are necessary here
const sycl::buffer<uint8_t, 2>& cv::GSYCLContext::inMat(int input)
{
    return (inArg<sycl::buffer<uint8_t, 2>>(input));
}

// FIXME: This will likely break util::get, may need to add support
sycl::buffer<uint8_t, 2>& cv::GSYCLContext::outMatR(int output)
{
    return (*(util::get<sycl::buffer<uint8_t, 2>*>(m_results.at(output))));
}

const cv::Scalar& cv::GSYCLContext::inVal(int input)
{
  return inArg<cv::Scalar>(input);
}

cv::Scalar& cv::GSYCLContext::outValR(int output)
{
  return *util::get<cv::Scalar*>(m_results.at(output));
}

cv::GSYCLKernel::GSYCLKernel()
{
}

cv::GSYCLKernel::GSYCLKernel(const GSYCLKernel::F &f)
  : m_f(f)
{
}

void cv::GSYCLKernel::apply(GSYCLContext &ctx)
{
  CV_Assert(m_f);
  m_f(ctx);
}
