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
 * \file moe_distribute_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine.h"
#include "moe_distribute_combine_tiling.h"
using namespace AscendC;
using namespace MoeDistributeCombineImpl;

namespace {
template <TemplateMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales,
                                                GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR XOut,
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<TemplateMC2TypeFunc> op;
    op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask,
        sharedExpertX, XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
}
}

/*
* A3 tilingkey说明
* 5位的十进制数
* 第1位（个位）：无意义占位使用
* 第2位（十位）：通信量化选项：
*     0：无量化, 2:int8量化
* 第3位（百位）：是否做tp域allgather:
*     0: 不做, 1: 做
* 第4位（千位）：是否是共享专家卡:
*     0: 不是, 1: 是
* 第5位（万位）：无实际意义
*/

extern "C" __global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                             GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                             GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                             GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR XOut,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineTilingData);
    TPipe pipe;
    GM_ADDR sharedExpertX = nullptr;

#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10100)) { // tp=2 isShared=0 IsInt8Quant=0
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, false, false>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10000)) { // tp=1 isShared=0 IsInt8Quant=0
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, false, false>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(11100)) { // tp=2 isShared=1 IsInt8Quant=0
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, true, false>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(11000)) { // tp=1 isShared=1 IsInt8Quant=0
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, true, false>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10120)) { // tp=2 isShared=0 IsInt8Quant=1
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, false, true>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10020)) { // tp=1 isShared=0 IsInt8Quant=1
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, false, true>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(11120)) { // tp=2 isShared=1 IsInt8Quant=1
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, true, true>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(11020)) { // tp=1 isShared=1 IsInt8Quant=1
        ExecMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, true, true>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM, &pipe);
    }
#endif
}