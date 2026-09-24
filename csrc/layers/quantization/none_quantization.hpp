#pragma once

#include "base_quantization.hpp"
#include <mutex>
#include <unordered_map>
namespace infinilm::quantization {

class NoneQuantization : public BaseQuantization {
public:
    explicit NoneQuantization(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config){};

    NoneQuantization();

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::NONE;
    };

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    infinicore::Tensor forward_allreduce(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        infinicclComm_t communicator,
        float alpha = 1.0f) const override;

    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    // With --pre-transpose, materialize [IC, OC] once. Eligible Ascend FP16/BF16
    // weights are then converted to FRACTAL_NZ; all other devices retain ND.
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params,
        const infinicore::Device &device,
        int split_dim = -1) const override;

private:
    enum class WeightLayout {
        NONE,          // checkpoint layout [OC, IC]
        ND_KN,         // pre-transposed contiguous [IC, OC]
        ASCEND_NZ_KN,  // logical [IC, OC], physical FRACTAL_NZ
    };

    WeightLayout weight_layout(const infinicore::Tensor &weight) const;

    // NoneQuantization is shared by multiple Linear layers. Layout therefore
    // belongs to the transformed weight tensor, not to this quantizer object.
    mutable std::mutex weight_layouts_mutex_;
    mutable std::unordered_map<const infinicore::TensorImpl *, WeightLayout>
        weight_layouts_;
};

} // namespace infinilm::quantization
