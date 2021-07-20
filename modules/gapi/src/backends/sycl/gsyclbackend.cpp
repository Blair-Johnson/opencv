#ifdef HAVE_SYCL

#include "precomp.hpp"

#include <ade/util/algorithm.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include <opencv2/core/base.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/sycl/gsyclbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

using GSYCLModel = ade::TypedGraph
  < cv::gimpl::SYCLUnit
  , cv::gimpl::Protocol
  >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGSYCLModel = ade::ConstTypedGraph
  < cv::gimpl::SYCLUnit
  , cv::gimpl::Protocol
  >;

namespace
{
  class GSYCLBackendImpl final: public cv::gapi::GBackend::Priv
  {
    virtual void unpackKernel(ade::Graph  &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
      GSYCLModel gm(graph);
      auto sycl_impl = cv::util::any_cast<cv::GSYCLKernel>(impl.opaque);
      gm.metadata(op_node).set(cv::gimpl::SYCLUnit{sycl_impl});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
      return EPtr{new cv::gimpl::GSYCLExecutable(graph, nodes)};
    }
  };
}

cv::gapi::GBackend cv::gapi::sycl::backend()
{
  static cv::gapi::GBackend this_backend(std::make_shared<GSYCLBackendImpl>());
  return this_backend;
}

// GSYCLExecutable implementation ////////////////////
cv::gimpl::GSYCLExecutable::GSYCLExecutable(const ade::Graph &g,
                                            const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
  // Convert op list into execution script
  // Possibly add interop node here to transition from Mats to sycl buffers
  for (auto &nh : nodes)
  {
    switch (m_gm.metadata(nh).get<NodeType>().t)
    {
      case NodeType::OP: m_script.push_back({nh, GModel::collectOutputMeta(m_gm, nh)}); break;
      case NodeType::DATA:
      {
        m_dataNodes.push_back(nh);
        const auto &desc = m_gm.metadata(nh).get<Data>();
        if (desc.storage == Data::Storage::CONST_VAL)
        {
          auto rc = RcDesc{desc.rc, desc.shape, desc.ctor};
          magazine::bindInArg(m_res, rc, m_gm.metadata(nh).get<ConstValue>().arg);
        }
        //This is where internal mat allocation happens
        //need to figure out how to use buffers instead
        if (desc.storage == Data::Storage::INTERNAL && desc.shape == GShape::GMAT)
        {
          const auto mat_desc = util::get<cv::GMatDesc>(desc.meta);
          //auto& mat = m_res.slot<cv::Mat>()[desc.rc];
          //createMat(mat_desc, mat);
        }
        break;
      }
      default: util::throw_error(std::logic_error("Unsupported NodeType type"));
    }
  }
}

// FIXME: Document what it does
cv::GArg cv::gimpl::GSYCLExecutable::packArg(const GArg& arg)
{
    // Verify that no API placeholders have made it into this scope
    // FIXME: Move this to compilation stage somewhere
    GAPI_Assert(arg.kind != cv::detail::ArgKind::GMAT &&
        arg.kind != cv::detail::ArgKind::GSCALAR &&
        arg.kind != cv::detail::ArgKind::GARRAY &&
        arg.kind != cv::detail::ArgKind::GOPAQUE);

    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other ases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU objects (either from host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc& ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT: return GArg(m_res.slot<sycl::buffer<uint8_t, 2>>()[ref.id]);
    case GShape::GSCALAR: return GArg(m_res.slot<cv::Scalar>()[ref.id]);
    case GShape::GARRAY: return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    case GShape::GOPAQUE: return GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GSYCLExecutable::run(std::vector<InObj>&& input_objs,
                                     std::vector<OutObj>&& output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    // TODO: Determine if cleaning up sycl::buffers is necessary as with cv::UMats
    // const auto clean_up = [&input_objs, &output_objs](cv::gimpl::Mag* p)
    // {
    // }

    const auto bindBuffer = [this](const RcDesc& rc)
    {   // FIXME: This is likely a broken implementation. Figure out how to perform
        //        type and dimension matching automatically
        auto& mag_buffer = m_res.template slot<sycl::buffer<uint8_t, 2>>()[rc.id];
        auto& mag_mat = m_res.template slot<cv::Mat>()[rc.id];
        CV_CheckTypeEQ(mag_mat.type(), CV_8UC1, "Only single channel UInt8 Mats "
                                    "are currently supported by SYCL");
        mag_buffer(mag_mat.data, range<2>(mag_mat.rows, mag_mat.cols));
    };

    for (auto& it : input_objs) {
        const auto& rc = it.first;
        magazine::bindInArg(m_res, rc, it.second);
        // There is already a cv::Mat in the magazine after bindInArg call,
        // extract buffer from it and put into magazine
        if (rc.shape == GShape::GMAT) bindBuffer(rc);
    }
    for (auto& it : output_objs) {
        const auto& rc = it.first;
        magazine::bindOutArg(m_res, rc, it.second);
        if (rc.shape == GShape::GMAT) bindBuffer(rc);
    }

    // Initialize (reset) internal data nodes with user structures
    // before processing a frame (no need to do it for external data structures)
    GModel::ConstGraph gm(m_g);
    for (auto nh : m_dataNodes)
    {
        const auto& desc = gm.metadata(nh).get<Data>();

        if (desc.storage == Data::Storage::INTERNAL &&
            !util::holds_alternative<util::monostate>(desc.ctor))
        {
            // FIXME: Note that compile-time constant data objects (like
            // a value-initialized GArray<T>) also satisfy this condition
            // and should be excluded, but now we just don't support it
            magazine::resetInternalData(m_res, desc);
        }
    }

    // Backend execution - invoke kernels in proper order.
    GConstGSYCLModel gcm(m_g);
    for (auto& op_info : m_script)
    {
        const auto& op = m_gm.metadata(op_info.nh).get<Op>();

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GSYCLKernel k = gcm.metadata(op_info.nh).get<SYCLUnit>().k;

        // Initialize kernel's execution context:
        // - Input parameters
        GSYCLContext context;
        context.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
            std::back_inserter(context.m_args),
            std::bind(&GSYCLExecutable::packArg, this, _1));

        // Output parameters.
        for (const auto out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused here?
            const auto out_port = ade::util::index(out_it);
            const auto& out_desc = ade::util::value(out_it);
            //context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc, true);
            switch (out_desc.shape)
            {
            case GShape::GMAT:
                context.m_results[out_port] = GRunArgP(&m_res.template slot<sycl::buffer<uint8_t, 2>>()[out_desc.id]);
            default:
                context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc, false);
                break;
            }
        }

        // Trigger executable unit
        k.apply(context);

        for (const auto out_it : ade::util::indexed(op_info.expected_out_metas))
        {
            const auto out_index = ade::util::index(out_it);
            const auto& expected_meta = ade::util::value(out_it);

            if (!can_describe(expected_meta, context.m_results[out_index]))
            {
                const auto out_meta = descr_of(context.m_results[out_index]);
                util::throw_error(std::logic_error("Output meta doesn't coincide with"
                    " the generated meta\n Expected: "
                    + ade::util::to_string(expected_meta) + "\n"
                    "Actual: " + ade::util::to_string(out_meta)));
            }
        }
    } // for(m_script)

    // FIXME: There may be a better place to do this
    // FIXME: Add exception handling for SYCL
    context.getQueue().wait_and_throw();

    // FIXME: Determine where back conversion from SYCL buffers to Mats occurs
    for (auto& it : output_objs)
    {
        const auto& rc = it.first;
        auto& g_arg = it.second;
        magazine::writeBack(m_res, rc, g_arg);
        if (rc.shape == GShape::GMAT)
        {
            // FIXME: Is this just a check or does this perform some assignment
            uchar* out_arg_data = m_res.template slot<cv::Mat>()[rc.id].data;
            //auto& mag_mat = m_res.template slot<cv::UMat>().at(rc.id);
            //GAPI_Assert((out_arg_data == (mag_mat.getMat(ACCESS_RW).data)) &&
            //    " data for output parameters was reallocated ?");
            auto& mag_buffer = m_res.template slot<sycl::buffer<uint8_t, 2>>().at(rc.id);

        }
    }

    // In/Out args clean-up is mandatory now with RMat
    // TODO: Check necessity with sycl
    for (auto& it : input_objs) magazine::unbind(m_res, it.first);
    for (auto& it : output_objs) magazine::unbind(m_res, it.first);
}

#endif // HAVE_SYCL
