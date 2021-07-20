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
};
}}

#endif // OPENCV_GAPI_GSYCLBACKEND_HPP

#endif // HAVE_SYCL
