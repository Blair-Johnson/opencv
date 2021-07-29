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

struct GAPI_EXPORTS RMatSYCLBufferAdapter final: public cv::RMat::Adapter
    {
        using MapDescF = std::function<getGMatDescFromSYCLDesc(SYCLBufferDesc&)>;
        //using MapDataF = std::function<cv::Mat(const

        template <typename T, int Dimensions, typename AllocatorT, typename Enable>
        RMatSYCLBufferAdapter(const sycl::buffer<T, Dimensions, AllocatorT, Enable>& buffer,
                              const MapDescF& bufferDescToMatDesc,
                              const MapDataF& bufferViewToMat):
          m_buffer(buffer),
          m_bufferDesc(getSYCLBufferDesc(buffer)),
          m_bufferDescToMatDesc(buffer),
          //m_bufferViewToMat(bufferViewToMat)
        { }

        virtual cv::RMat::View access(cv::RMat::Access a, sycl::queue& queue) override
        {
            template <typename T, typename H>
            auto rmatToSYCLAccess = [](cv::RMat::Access rmatAccess){
                switch(rmatAccess) {
                    case cv::RMat::Access::R:
                        return std::function<sycl::host_accessor{T&}>;
                    case cv::RMat::Access::W:
                        return std::function<sycl::accessor(T, H)>;
                    default:
                        cv::util::throw_error(std::logic_error("Only cv::RMat::Access::R"
                              " or cv::RMat::Access::W can be mapped to sycl::accessors"));
                }
            }

            // FIXME: Need to think though how to accomplish this
            auto fv = rmatToSYCLAccess(a)(m_buffer); // create accessor or host accessor

            auto fvHolder = std::make_shared<>(std::move(fv));
            auto callback = [fvHolder]() mutable { fvHolder.reset(); };

            return asView(m_bufferViewToMat(m_frame.desc(), *fvHolder), callback);
        }

        virtual cv::GMatDesc desc() const override
        {
            return m_bufferDescToMatDesc(m_bufferDesc);
        }

        template <typename T, int Dimensions, typename AllocatorT, typename Enable>
        sycl::buffer<T, Dimensions, AllocatorT, Enable> m_buffer;
        SYCLBufferDesc m_bufferDesc;
        MapDescF m_bufferDescToMatDesc;
        MapDataF m_bufferViewToMat;
    };
} // namespace gimpl
} // namespace cv



#endif // OPENCV_GAPI_GSYCLBACKEND_HPP

#endif // HAVE_SYCL
