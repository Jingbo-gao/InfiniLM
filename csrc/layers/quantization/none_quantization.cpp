#include "none_quantization.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops/ascend_format_cast.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_allreduce.hpp"
#include <optional>

namespace infinilm::quantization {

NoneQuantization::NoneQuantization() : NoneQuantization(nlohmann::json()) {}

NoneQuantization::WeightLayout NoneQuantization::weight_layout(
    const infinicore::Tensor &weight) const {
    std::lock_guard<std::mutex> lock(weight_layouts_mutex_);
    auto it = weight_layouts_.find(weight.operator->());
    return it == weight_layouts_.end() ? WeightLayout::NONE : it->second;
}

std::vector<ParamDescriptor> NoneQuantization::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, dtype, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim >= 0 ? 0 : -1, split_dim >= 0 ? tp_rank : 0, split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor NoneQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    const auto layout = weight_layout(weight);
    if (layout == WeightLayout::ASCEND_NZ_KN) {
        return infinicore::op::linear_packed_nz(
            input_contiguous, weight, bias_opt, alpha);
    }
    if (layout == WeightLayout::ND_KN) {
        return infinicore::op::linear_packed(input_contiguous, weight, bias_opt, alpha);
    }
    return infinicore::op::linear(input_contiguous->contiguous(), weight->contiguous(), bias_opt, alpha);
}

infinicore::Tensor NoneQuantization::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    if (alpha != 1.0f) {
        return BaseQuantization::forward_allreduce(
            params, input, has_bias, communicator, alpha);
    }

    auto input_contiguous = input->is_contiguous()
                              ? input
                              : input->contiguous();
    auto weight = params.at("weight");
    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    const auto layout = weight_layout(weight);
    if (layout == WeightLayout::ASCEND_NZ_KN) {
        return infinicore::op::linear_allreduce_packed_nz(
            input_contiguous, weight, bias_opt, communicator);
    }
    if (layout == WeightLayout::ND_KN) {
        return infinicore::op::linear_allreduce_packed(
            input_contiguous, weight, bias_opt, communicator);
    }
    return infinicore::op::linear_allreduce(
        input_contiguous, weight->contiguous(), bias_opt, communicator);
}

std::vector<SplitParam> NoneQuantization::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto bias_it = params.find("bias");

    // The offsets in `splits` address the output dim (OC) of the checkpoint
    // [OC, IC] layout. After pre-transpose / NZ conversion the weight becomes
    // [IC, OC], so the same OC offsets live on the transposed dim. Flip the
    // narrow dim accordingly, otherwise the dim-0 bound check fails at small
    // tp sizes (e.g. gate_up up-proj at tp=4 needs start+len=14784 > 8192).
    int weight_dim = narrow_dim;
    const auto layout = weight_layout(weight_it->second);
    if (layout == WeightLayout::ND_KN || layout == WeightLayout::ASCEND_NZ_KN) {
        weight_dim = (narrow_dim == 0) ? 1 : 0;
    }

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight_it->second->narrow({{static_cast<size_t>(weight_dim), s.start, s.size}}),
                              weight_dim, tp_rank, tp_size, s.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> NoneQuantization::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int split_dim) const {

    // Controlled by --pre-transpose CLI flag, default off.
    if (!global_state::get_infinilm_config().pre_transpose) {
        return nullptr;
    }

    auto weight_it = params.find("weight");
    if (weight_it != params.end()) {
        // A repeated hook invocation for this exact parameter is already done.
        if (weight_layout(weight_it->second) != WeightLayout::NONE) {
            return std::const_pointer_cast<BaseQuantization>(shared_from_this());
        }

        // Transpose weight from [OC, IC] to [IC, OC] once.
        // contiguous() materializes the transposed layout so that
        // subsequent forwards can feed it directly to GEMM.
        auto packed_nd = weight_it->second->permute({1, 0})->contiguous();

        const bool nz_eligible =
            device.getType() == infinicore::Device::Type::ASCEND
            // Row-parallel layers use the fused MatmulAllReduce API, which has
            // no WeightNz variant in the installed CANN headers. Keep that
            // path on ND. Bias is supported by linear_packed_nz_ as
            // MatmulWeightNz followed by an Ascend in-place broadcast add.
            && split_dim != 1
            && packed_nd->ndim() == 2
            && (packed_nd->dtype() == infinicore::DataType::F16
                || packed_nd->dtype() == infinicore::DataType::BF16)
            && packed_nd->size(0) % 16 == 0
            && packed_nd->size(1) % 16 == 0;

        if (nz_eligible) {
            params["weight"] = infinicore::op::ascend_format_cast_nz(packed_nd);
            std::lock_guard<std::mutex> lock(weight_layouts_mutex_);
            weight_layouts_[params["weight"].operator->()] =
                WeightLayout::ASCEND_NZ_KN;
        } else {
            params["weight"] = packed_nd;
            std::lock_guard<std::mutex> lock(weight_layouts_mutex_);
            weight_layouts_[params["weight"].operator->()] =
                WeightLayout::ND_KN;
        }
    }

    // Must return non-null so that BaseLinear::process_weights_after_loading
    // writes the modified params back into parameters_.
    // Returning shared_from_this() triggers the "quantization changed" path
    // which calls parameters_.clear() + re-insert from params.
    return std::const_pointer_cast<BaseQuantization>(shared_from_this());
}

} // namespace infinilm::quantization
