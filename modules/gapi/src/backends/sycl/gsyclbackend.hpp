#ifdef HAVE_SYCL

#ifndef OPENCV_GAPI_GSYCLBACKEND_HPP
#define OPENCV_GAPI_GSYCLBACKEND_HPP

#include <map>
#include <unordered_map>
#include <tuple>
#include <ade/util/algorithm.hpp>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/sycl/gsyclkernel.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

#include <CL/sycl.hpp>

namespace cv { namespace gimpl {

struct SYCLUnit
{
  static const char *name() { return "SYCLKernel"; }
  GSYCLKernel k;
};

class GSYCLExecutable final: public GIslandExecutable
{
    const ade::Graph& m_g;
    GModel::ConstGraph m_gm;

    struct OperationInfo
    {
        ade::NodeHandle nh;
        GMetaArgs expected_out_metas;
    };

    // Execution script, currently naive
    std::vector<OperationInfo> m_script;
    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;

    // Actual data of all resources in graph (both internal and external)
    Mag m_res;
    GArg packArg(const GArg& arg);
    sycl::queue m_queue;

public:
    GSYCLExecutable(const ade::Graph  &graph,
                    const std::vector<ade::NodeHandle> &nodes);

    // FIXME: Can this be made reshapable?
    virtual inline bool canReshape() const override { return false; }
    inline void reshape(ade::Graph&, const GCompileArgs&) override
    {
        util::throw_error(std::logic_error("GSYCLExecutable::reshape() should never be called"));
    }

    virtual void run(std::vector<InObj> &&input_objs,
                     std::vector<OutObj> &&ouput_objs) override;

    void initSYCLContext();
};

template <typename T>
int syclTypeToMatType()
    {
        if (std::is_same<T, unsigned char>::value){
            return CV_8U;
        }
        else if (std::is_same<T, signed char>::value){
            return CV_8S;
        }
        else if (std::is_same<T, unsigned int>::value){
            return CV_16U;
        }
        else if (std::is_same<T, int>::value){
            return CV_16S;
        }
        else if (std::is_same<T, long int>::value){
            return CV_32S;
        }
        else if (std::is_same<T, float>::value){
            return CV_32F;
        }
        else if (std::is_same<T, double>::value){
            return CV_64F;
        }
    };

struct GAPI_EXPORTS SYCLBufferDesc
    {
        int depth;
        int chan;
        cv::Size size;
        bool planar;
    };

template <typename T, int Dimensions, typename AllocatorT, typename Enable>
SYCLBufferDesc getSYCLBufferDesc(sycl::buffer<T, Dimensions, AllocatorT, Enable>& buffer)
    {
        SYCLBufferDesc desc;
        desc.depth = sizeof(T)*8; // size in bits
        desc.chan = buffer.get_range().get(2); // assume channel dim is third
        desc.size = cv::Size(buffer.get_range().get(0), buffer.get_range().get(1));
        if (desc.chan == 1)
        {
            desc.planar = true;
        }
        else
        {
            desc.planar = false;
        }
        return desc;
    };

cv::GMatDesc getGMatDescFromSYCLDesc(SYCLBufferDesc& desc)
    {
        return cv::GMatDesc(desc.depth, desc.chan, desc.size, desc.planar);
    };

template <typename T, int Dimensions, typename AllocatorT, typename Enable>
struct GAPI_EXPORTS RMatSYCLBufferAdapter final: public cv::RMat::Adapter
    {
        using ReMapDescF = std::function<cv::GMatDesc(const SYCLBufferDesc&)>;

        RMatSYCLBufferAdapter(const sycl::buffer<T, Dimensions, AllocatorT, Enable>& buffer,
                              const ReMapDescF& bufferDescToGMatDesc):
          m_buffer(buffer),
          m_bufferDesc(getSYCLBufferDesc(buffer)),
          m_bufferDescToGMatDesc(bufferDescToGMatDesc)
        { }

        virtual cv::RMat::View access(cv::RMat::Access a) override
        {
            // Map RMat access flag to sycl access mode Tag
            auto rmatToSYCLAccess = [](cv::RMat::Access rmatAccess){
                switch(rmatAccess) {
                    case cv::RMat::Access::R:
                        return sycl::access_mode::read;
                    case cv::RMat::Access::W:
                        return sycl::access_mode::write;
                    //case cv::RMat::Access::RW:
                    //    return sycl::access_mode::read_write;
                    //    not supported by RMat currently
                    default:
                        cv::util::throw_error(std::logic_error("Only cv::RMat::Access::R"
                              " or cv::RMat::Access::W can be mapped to sycl::accessors"));
                }
            };

            // Make sycl host accessor with buffer reference and sycl access flag
            sycl::host_accessor syclHostAccessor(m_buffer, rmatToSYCLAccess(a));
            // create shared ptr to sycl host accessor
            auto syclSharedAccessor = std::make_shared<sycl::host_accessor>(syclHostAccessor);
            // create callback function to trigger accessor destructor on RMat::View destruction
            auto callback = [syclSharedAccessor]{ syclSharedAccessor.reset(); };

            // FIXME: Figure out how to map sycl buffer metadata to cv types
            return asView(cv::Mat(m_bufferDesc.size, syclTypeToMatType<T>(),
                          syclHostAccessor.get_pointer()),  callback);
        }

        virtual cv::GMatDesc desc() const override
        {
            return m_bufferDescToGMatDesc(m_bufferDesc);
        }

        sycl::buffer<T, Dimensions, AllocatorT, Enable>& m_buffer;
        SYCLBufferDesc m_bufferDesc;
        ReMapDescF m_bufferDescToGMatDesc;
    };
} // namespace gimpl
} // namespace cv



#endif // OPENCV_GAPI_GSYCLBACKEND_HPP

#endif // HAVE_SYCL
