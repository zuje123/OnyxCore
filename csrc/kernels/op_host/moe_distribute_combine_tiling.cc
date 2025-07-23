/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_combine__tiling.cc
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

//#include "runtime/base/mc2/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
//#include "op_log.h"
//#include "op_util.h"
//#include "hcom/hcom_topo_info.h"
//#include "op_log.h"
#include "error_log.h"
//#include "error_util.h"
//#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
//#include "platform/platform_infos_def.h"
#include "../op_kernel/moe_distribute_combine_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
        class Mc2TilingUtils {
    public:
        #define HCCL_BUFFSIZE  "HCCL_BUFFSIZE"
        static uint64_t GetMaxWindowSize()
        {
            uint16_t defaultWindowSize = 200;
            if (getenv(HCCL_BUFFSIZE) == nullptr) {
                OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
            } else {
                try {
                    std::string envStr(getenv(HCCL_BUFFSIZE));
                    defaultWindowSize = std::stoi(envStr);
                } catch (...) {
                    OP_LOGE("", "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
                }
            }
            const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
            OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
            return maxWindowSize;
        }
    };
    constexpr uint32_t EXPAND_X_INDEX = 0;
    constexpr uint32_t EXPERT_IDS_INDEX = 1;
    constexpr uint32_t ASSIST_INFO_INDEX = 2;
    constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
    constexpr uint32_t EXPERT_SCALES_INDEX = 4;
    constexpr uint32_t TP_SEND_COUNTS_INDEX = 5;
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 6;
    constexpr uint32_t ACTIVATION_SCALE_INDEX = 7;
    constexpr uint32_t WEIGHT_SCALE_INDEX = 8;
    constexpr uint32_t GROUP_LIST_INDEX = 9;
    // constexpr uint32_t SHARED_EXPERT_X_INDEX = 11;
    constexpr uint32_t OUTPUT_X_INDEX = 0;

    constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
    constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
    constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
    constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
    constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
    constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
    constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
    constexpr uint32_t ATTR_OUT_DTYPE_INDEX = 11;
    constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;
    constexpr uint32_t ATTR_GROUP_LIST_TYPE_INDEX = 13;

    constexpr uint32_t INT8_COMM_QUANT = 2U;
    constexpr uint64_t INIT_TILINGKEY = 10000;
    constexpr uint64_t TILINGKEY_TP_WORLD_SIZE = 100;
    constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
    constexpr uint64_t TILINGKEY_IS_SHARE_EXPERT = 1000;
    constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

    constexpr uint32_t THREE_DIMS = 3U;
    constexpr uint32_t TWO_DIMS = 2U;
    constexpr uint32_t ONE_DIM = 1U;
    constexpr uint32_t ASSIST_INFO_DIMS = 1U;
    constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
    constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
    constexpr uint64_t TILING_KEY_INT8_COMM_QUANT_A2 = 100UL;
    constexpr uint32_t ARR_LENGTH = 128U;
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U; // numeric representation of AlltoAll
    constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll

    constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
    constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
    constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
    constexpr uint32_t MAX_BATCH_SIZE_LAYERED_A2 = 128;
    constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
    constexpr uint32_t BLOCK_SIZE_A2 = 32;
    constexpr uint32_t K_VALUE_A2 = 8;
    constexpr uint32_t MAX_K_VALUE_A2 = 16;
    constexpr uint32_t LAYERED_SUPPORT_K = 8;
    constexpr uint32_t LAYERED_SUPPORT_K_MAX = 16;
    constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
    constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
    constexpr int64_t MAX_EP_WORLD_SIZE = 384;
    constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    constexpr int64_t EP_RESTRICT_8 = 8;
    constexpr int64_t MAX_TP_WORLD_SIZE = 2;
    constexpr int64_t BS_UPPER_BOUND = 512;

    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
    constexpr uint32_t VERSION_2 = 2;
    constexpr uint32_t HCOMMCNT_2 = 2;
    constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
    constexpr int64_t K_MAX = 16;
    constexpr int64_t H_MIN = 1024;
    constexpr int64_t H_MAX = 7168;
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr uint64_t TRIPLE = 3;
    constexpr uint64_t ASSIST_NUM_PER_A = 128UL;
    constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL; // scale32B + 3*4expandIdx
    constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
    constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
    constexpr uint64_t UB_ALIGN = 32UL;
    constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;

    enum class CommQuantMode : int32_t {
        NON_QUANT = 0,
        INT12_QUANT = 1,
        INT8_QUANT = 2
    };
    using CommQuantModeType = std::underlying_type<CommQuantMode>;
}

namespace optiling {

// a3专有
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineTilingData& tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineInfo.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeCombineInfo.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineInfo.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineInfo.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineInfo.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeCombineInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeCombineInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeCombineInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeCombineInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeCombineInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.moeDistributeCombineInfo.totalWinSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, std::string &groupEp, std::string &groupTp, uint32_t &commQuantMode)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    // 判空
    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH), OP_LOGE(nodeName, "groupEp is invalid."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardType is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE(nodeName, "commQuantMode is null."), return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*commQuantModePtr != 0), OP_LOGE(nodeName,
            "commQuantMode only supports 0 when tpWorldSize > 1, but got commQuantMode=%ld, tpWorldSize=%ld.",
            *commQuantModePtr, *tpWorldSizePtr), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
        MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
        epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, moeExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*commQuantModePtr != 0) && (*commQuantModePtr != INT8_COMM_QUANT),
        OP_LOGE(nodeName, "commQuantMode only support 0(default) or 2(int8 comm quant), but got commQuantMode=%ld.",
        *commQuantModePtr), return ge::GRAPH_FAILED);
    int64_t moePerRankNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    int64_t curDispatchStatusNum = moePerRankNum * epWorldSize;
    OP_TILING_CHECK((curDispatchStatusNum > DISPATCH_STATUS_MAX_SUPPORT_NUM),
        OP_LOGE(nodeName, "The moe experts num must meet the conditions,"
        " (moeExpertNum / (epWorldSize - sharedExpertRankNum)) * epWorldSize <= 1280, but cur is %ld.",
        curDispatchStatusNum), return ge::GRAPH_FAILED);

    commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    groupEp = string(groupEpPtr);
    tilingData.moeDistributeCombineInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeCombineInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeCombineInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeCombineInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeCombineInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeCombineInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeCombineInfo.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeCombineInfo.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeCombineInfo.sharedExpertNum == 1U) {
            tilingData.moeDistributeCombineInfo.sharedExpertNum = 0U;
        }
    }
    tilingData.moeDistributeCombineInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static bool CheckInputTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandX is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandX must be 2-dimension, but got %lu dim",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIds is null."), return false);
    OP_TILING_CHECK(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIds must be 2-dimension, but got %lu dim",
        expertIdsStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsDim0);
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsDim1);

    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoForCombine is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "assistInfoForCombine must be 1-dimension, but got %lu dim",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."), return false);
    OP_TILING_CHECK(epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epSendCounts must be 1-dimension, but got %lu dim",
        epSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."), return false);
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertScales must be 2-dimension, but got %lu dim",
        expertScalesStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));

    return true;
}

static bool CheckOptionalInputTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask)
{
    const gert::StorageShape *tpSendCountsStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "tpSendCounts is null."), return false);
    OP_TILING_CHECK(tpSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "tpSendCounts must be 1-dimension, but got %lu dim",
        tpSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "tpSendCounts dim0 = %ld", tpSendCountsStorageShape->GetStorageShape().GetDim(0));

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask is null."), return false);
        OP_TILING_CHECK(xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "xActiveMask must be 1-dimension, but got %lu dim",
            xActiveMaskStorageShape->GetStorageShape().GetDimNum()), return false);
    }

    const gert::StorageShape *activationScaleStorageShape = context->GetOptionalInputShape(ACTIVATION_SCALE_INDEX);
    OP_TILING_CHECK(activationScaleStorageShape != nullptr, OP_LOGE(nodeName, "activationScale is not null."), return false);

    const gert::StorageShape *weightScaleStorageShape = context->GetOptionalInputShape(WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(weightScaleStorageShape != nullptr, OP_LOGE(nodeName, "weightScale is not null."), return false);
    
    const gert::StorageShape *groupListStorageShape = context->GetOptionalInputShape(GROUP_LIST_INDEX);
    OP_TILING_CHECK(groupListStorageShape != nullptr, OP_LOGE(nodeName, "groupList is not null."), return false);
    
    // const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    // if (sharedExpertX != nullptr) {
    //     auto attrs = context->GetAttrs();
    //     auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    //     OP_TILING_CHECK(*sharedExpertRankNumPtr != 0, OP_LOGE(nodeName, "sharedExpertX only support input None "\
    //         "when sharedExpertRankNum is non-zero."), return false);
    //     OP_TILING_CHECK(((sharedExpertX->GetStorageShape().GetDimNum() != TWO_DIMS) &&
    //                     (sharedExpertX->GetStorageShape().GetDimNum() != THREE_DIMS)),
    //                     OP_LOGE(nodeName, "sharedExpertX must be 2-dimension or 3-dimension, but got %lu dim",
    //                             sharedExpertX->GetStorageShape().GetDimNum()), return false);
    // }

    return true;
}

static bool CheckOutputTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "x must be 2-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    return true;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, isActiveMask),
        OP_LOGE(nodeName, "param shape of input tensor is invalid"), return false);
    
    OP_TILING_CHECK(!CheckOptionalInputTensorDim(context, nodeName, isActiveMask),
        OP_LOGE(nodeName, "param shape of optional input tensor is invalid"), return false);
    
    OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName, isActiveMask),
        OP_LOGE(nodeName, "param shape of output tensor is invalid"), return false);

    return true;
}

// 校验数据类型
static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName, const bool isActiveMask)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is "
        ), return false);
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32), OP_LOGE(nodeName, "expertIds dataType is invalid, "
        "dataType should be int32, but is " ), return false);
    auto assistInfoDesc = context->GetInputDesc(ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK((assistInfoDesc->GetDataType() != ge::DT_INT32), OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid,"
        " dataType should be int32, but is"), return false);
    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "epSendCounts dataType is invalid, dataType should be int32, but is "),
        return false);
    auto tpSendCountsDesc = context->GetOptionalInputDesc(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
    OP_TILING_CHECK((tpSendCountsDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "tpSendCounts dataType is invalid, dataType should be int32, but is "), return false);
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(nodeName, "xActiveMask dataType is invalid,"
            " dataType should be bool, but is"), return false);
    }
    // auto sharedExpertXDesc = context->GetOptionalInputDesc(SHARED_EXPERT_X_INDEX);
    // if (sharedExpertXDesc != nullptr) {
    //     OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
    //         OP_LOGE(nodeName, "sharedExpertX dataType should be the same as expandX dataType, but got sharedExpertX"
    //         "dataType , expandX dataType ."), return false);
    // }
    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(nodeName, "expertScales dataType is invalid, dataType should be float, but is "),
         return false);
    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != expandXDesc->GetDataType()), OP_LOGE(nodeName,
        "x dataType is invalid, dataType should be equal to expandX dataType , but is "),
        return false);
    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName, const bool isActiveMask)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandXFormat is invalid"), return false);

    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertIdsFormat is invalid"), return false);

    auto assistInfoDesc = context->GetInputDesc(ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "assistInfoFormat is invalid"), return false);

    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epSendCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epSendCountsFormat is invalid"), return false);

    auto tpSendCountsDesc = context->GetOptionalInputDesc(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpSendCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpSendCountsFormat is invalid"), return false);

    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertScalesFormat is invalid"), return false);

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "xActiveMaskFormat is invalid."), return false);
    }

    // auto sharedExpertXDesc = context->GetOptionalInputDesc(SHARED_EXPERT_X_INDEX);
    // OP_TILING_CHECK((sharedExpertXDesc != nullptr) &&
    //     (static_cast<ge::Format>(ge::GetPrimaryFormat(sharedExpertXDesc->GetStorageFormat())) ==
    //     ge::FORMAT_FRACTAL_NZ), OP_LOGE(nodeName, "sharedExpertXFormat is invalid."), return false);

    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "xFormat is invalid"), return false);
    
    return true;
}

static bool CheckTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, bool isShared, uint32_t localExpertNum)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.moeExpertNum);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX || (expertIdsDim1 > moeExpertNum)),
        OP_LOGE(nodeName, "expertIds's dim1(K) should be in (0, min(%ld, moeExpetNum %ld)], "
        "but got expertIds's dim1=%ld.", K_MAX, moeExpertNum, expertIdsDim1), return false);
    tilingData.moeDistributeCombineInfo.k = static_cast<uint32_t>(expertIdsDim1);
 
    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeCombineInfo.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;
    if (isShared) { // 本卡为共享专家
        uint32_t rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        uint32_t epWorldSizeU32 = tilingData.moeDistributeCombineInfo.epWorldSize;
        uint32_t maxBs = globalBs / epWorldSizeU32;
        uint32_t maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
        A = maxBs * maxSharedGroupNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localExpertNum), expertIdsDim1);
    }

    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A * tpWorldSize, expandX's dim0 = %ld, A = %ld, tpWorldSize = %ld",
        expandXDim0, static_cast<int64_t>(A), tpWorldSize), return false);
    OP_TILING_CHECK((expandXDim1 < H_MIN) || (expandXDim1 > H_MAX),
        OP_LOGE(nodeName, "expandX's dim1(H) should be in [%ld, %ld], but got %ld.",
        H_MIN, H_MAX, expandXDim1), return false); // 32对齐
    tilingData.moeDistributeCombineInfo.h = static_cast<uint32_t>(expandXDim1);

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(ASSIST_INFO_INDEX);
    int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
//    OP_TILING_CHECK(assistInfoDim0 < static_cast<int64_t>(A * ASSIST_NUM_PER_A), OP_LOGE(nodeName,
//        "assistInfoForCombine's dim0 < A * 128, assistInfoForCombine's dim0 is %ld, A * 128 is %ld.", assistInfoDim0, static_cast<int64_t>(A * ASSIST_NUM_PER_A)),
//        return false);

    // 校验epSendCount和tpSendCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    const gert::StorageShape *tpSendCountStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpSendCountDim0 = tpSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t localEpSendCountSize = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;
    OP_TILING_CHECK(epSendCountDim0 < localEpSendCountSize * tpWorldSize, OP_LOGE(nodeName,
        "epSendCount's dim0 not greater than or equal to localEpSendCountSize * tpWorldSize, "
        "epSendCount's dim0 is %ld, localEpSendCountSize is %ld, tpWorldSize is %ld.",
        epSendCountDim0, localEpSendCountSize, tpWorldSize), return false);
    OP_TILING_CHECK(tpSendCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "tpSendCount's dim0 not equal to tpWorldSize, tpSendCount's dim0 is %ld, tpWorldSize is %ld.",
        tpSendCountDim0, tpWorldSize), return false);

    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != expertIdsDim0,
        OP_LOGE(nodeName, "expertScales's dim0 not equal to bs, expertScales's dim0 = %ld, bs = %ld",
        expertScalesDim0, expertIdsDim0), return false);
    OP_TILING_CHECK(expertScalesDim1 != expertIdsDim1, OP_LOGE(nodeName,
        "expertScales's dim1 not equal to k, expertScales's dim1 = %ld, k = %ld",
        expertScalesDim1, expertIdsDim1), return false);

    // 校验activeMask的维度
    if (tilingData.moeDistributeCombineInfo.isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != expertIdsDim0,
            OP_LOGE(nodeName, "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
            "expertIds's dim0 is %ld", xActiveMaskDim0, expertIdsDim0), return false);
    }

    tilingData.moeDistributeCombineInfo.hasSharedExpertX = false;
    // 校验sharedExpertX的维度
    // const gert::StorageShape *sharedExpertXShape = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    // tilingData.moeDistributeCombineInfo.hasSharedExpertX = (sharedExpertXShape != nullptr);
    // if (sharedExpertXShape != nullptr) {
    //     int64_t sharedExpertXDim0 = sharedExpertXShape->GetStorageShape().GetDim(0);
    //     int64_t sharedExpertXDim1 = sharedExpertXShape->GetStorageShape().GetDim(1);
    //     if (sharedExpertXShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
    //         OP_TILING_CHECK(sharedExpertXDim0 != expertIdsDim0,
    //             OP_LOGE(nodeName, "sharedExpertX's dim0 not equal to bs, sharedExpertX's dim0 = %ld, bs = %ld",
    //             sharedExpertXDim0, expertIdsDim0), return false);
    //         OP_TILING_CHECK(sharedExpertXDim1 != expandXDim1, OP_LOGE(nodeName,
    //             "sharedExpertX's dim1 not equal to h, sharedExpertX's dim1 = %ld, h = %ld",
    //             sharedExpertXDim1, expandXDim1), return false);
    //     } else {
    //         int64_t sharedExpertXDim2 = sharedExpertXShape->GetStorageShape().GetDim(TWO_DIMS);
    //         OP_TILING_CHECK(sharedExpertXDim0 * sharedExpertXDim1 != expertIdsDim0,
    //             OP_LOGE(nodeName, "sharedExpertX's dim0 * sharedExpertX's dim1 not equal to bs, sharedExpertX's dim0 * sharedExpertX's dim1 = %ld, bs = %ld",
    //             sharedExpertXDim0 * sharedExpertXDim1, expertIdsDim0), return false);
    //         OP_TILING_CHECK(sharedExpertXDim2 != expandXDim1, OP_LOGE(nodeName,
    //             "sharedExpertX's dim2 not equal to h, sharedExpertX's dim2 = %ld, h = %ld",
    //             sharedExpertXDim2, expandXDim1), return false);
    //     }
    // }

    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName,
        "x's dim0 not equal to bs, bs = %ld, x's dim0 = %ld", expertIdsDim0, xDim0), return false);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName,
        "x's dim1 not equal to h, x's dim1 = %ld, h = %ld", xDim1, expandXDim1), return false);

    return true;
}

static bool CheckSharedAttrs(gert::TilingContext *context, const char *nodeName,
    MoeDistributeCombineTilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support 0 when sharedExpertNum is 0, but got %u.",
        sharedExpertRankNum), return false);
    OP_TILING_CHECK((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 when sharedExpertRankNum is 0, but got %u.",
        sharedExpertNum), return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName, "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
            "sharedExpertNum=%u.", sharedExpertRankNum, sharedExpertNum), return false);
    }

    return true;
}

static bool CheckAttrs(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeCombineInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(context, nodeName, tilingData),
        OP_LOGE(nodeName, "Check shared expert related attributes falied."), return false);

    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but got moeExpertNum=%d, epWorldSize=%d, sharedExpertRankNum=%d.", moeExpertNum, epWorldSize,
        sharedExpertRankNum), return false);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
        OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %d", localMoeExpertNum), return false);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((localMoeExpertNum > 1) && (tpWorldSize > 1),
        OP_LOGE(nodeName, "Cannot support multi-moeExpert %d in a rank when tpWorldSize = %d > 1",
        localMoeExpertNum, tpWorldSize), return false);
    tilingData.moeDistributeCombineInfo.moeExpertPerRankNum = localMoeExpertNum;

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > BS_UPPER_BOUND),
        OP_LOGE(nodeName, "Invalid expertIds dims0(BS) %ld. Should be between [1, %ld].",
        expertIdsDim0, BS_UPPER_BOUND), return false);
    tilingData.moeDistributeCombineInfo.bs = static_cast<uint32_t>(expertIdsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    bool isActiveMask = tilingData.moeDistributeCombineInfo.isActiveMask;
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    OP_LOGD(nodeName, "MoeDistributeCombine *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, expertIdsDim0, epWorldSize);

    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize),  return false);
    OP_TILING_CHECK(((*globalBsPtr > (expertIdsDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize), return false);
    
    tilingData.moeDistributeCombineInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineInfo.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    }

    return true;
}

static ge::graphStatus TilingCheckMoeDistributeCombine(gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isActiveMask),
                    OP_LOGE(nodeName, "param shape is invalid"), return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isActiveMask),
                    OP_LOGE(nodeName, "param dataType is invalid"), return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isActiveMask),
                    OP_LOGE(nodeName, "param Format is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspce[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

static void CalTilingKey(uint64_t &tilingKey, const uint64_t tpWorldSize, const bool isShared, uint32_t commQuantMode)
{
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingKey += TILINGKEY_TP_WORLD_SIZE;
    }
    if (isShared) {
        tilingKey += TILINGKEY_IS_SHARE_EXPERT;
    }
    if (commQuantMode == INT8_COMM_QUANT) {
        tilingKey += TILINGKEY_INT8_COMM_QUANT;
    }
}

static void SetHCommCfg(gert::TilingContext *context, MoeDistributeCombineTilingData *tiling,
    const std::string groupEp, const std::string groupTp)
{
    const char* nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombine groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfigReduceScatterStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2);
}

static ge::graphStatus MoeDistributeCombineA3TilingFuncImpl(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeCombine Tiling func");
    MoeDistributeCombineTilingData *tilingData = context->GetTilingData<MoeDistributeCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    bool isShared = true;
    uint32_t localMoeExpertNum = 1;
    bool isActiveMask = false;
    uint32_t commQuantMode = 0U;

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp, commQuantMode) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeCombineInfo.isActiveMask = isActiveMask;
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckMoeDistributeCombine(context, nodeName, isActiveMask) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum),
        OP_LOGE(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertNum = tilingData->moeDistributeCombineInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineInfo.sharedExpertRankNum;
    uint32_t epRankId = tilingData->moeDistributeCombineInfo.epRankId;
    uint32_t localExpertNum;
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isShared = false;
        localExpertNum = localMoeExpertNum;
    } else { // 本卡为共享专家
        localExpertNum = 1U;
    }

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, isShared, localExpertNum),
        OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.epWorldSize);
    uint64_t k = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.k);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.globalBs)/ epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE  + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen = ((h * MAX_OUT_DTYPE_SIZE  + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize = ((maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum))
        + (maxBs * tokenNeedSizeCombine * (k + static_cast<uint64_t>(sharedExpertNum)))) * DOUBLE_DATA_BUFFER;
    OP_TILING_CHECK((actualSize > maxWindowSize),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, sharedExpertNum = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum) +"
            " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB,"
            " HCCL_BUFFSIZE=%luMB.", maxBs, h, epWorldSize, localMoeExpertNum, sharedExpertNum,
            tokenNeedSizeDispatch, tokenNeedSizeCombine, k, actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE),
            return ge::GRAPH_FAILED);
    tilingData->moeDistributeCombineInfo.totalWinSize = maxWindowSize;

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    SetHCommCfg(context, tilingData, groupEp, groupTp);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.tpWorldSize);
    uint64_t tilingKey = INIT_TILINGKEY;
    CalTilingKey(tilingKey, tpWorldSize, isShared, commQuantMode);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t blockDim = 1U;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData->moeDistributeCombineInfo.aivNum = aivNum;
    tilingData->moeDistributeCombineInfo.totalUbSize = ubSize;
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName, "blockdim = %u, aivNum = %lu, ubsize = %lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineTilingFunc(gert::TilingContext* context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is "),
                     return ge::GRAPH_FAILED);

    ge::graphStatus ret = MoeDistributeCombineA3TilingFuncImpl(context);
    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombine)
    .Tiling(MoeDistributeCombineTilingFunc)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombine);
} // namespace optiling