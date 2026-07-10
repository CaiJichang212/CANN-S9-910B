/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_def.cpp
 * \brief SquareSumV1 operator definition
 *
 * aclnnSquareSumV1(input, axis, keep_dims, result)
 * - input: FLOAT16 / BFLOAT16 / FLOAT, 1-5D
 * - axis: INT64 array (attribute)
 * - keep_dims: BOOL (attribute)
 * - result: same dtype as input, shape determined by axis + keep_dims
 */

#include "register/op_def_registry.h"

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("axis")
            .AttrType(REQUIRED)
            .ListInt();

        this->Attr("keep_dims")
            .AttrType(OPTIONAL)
            .Bool(false);

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);

        this->AICore().AddConfig("ascend910_93", aicoreConfig910B);
    }
};
OP_ADD(SquareSumV1);
} // namespace ops
