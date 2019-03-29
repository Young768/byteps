// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef BYTEPS_MXNET_MPI_OPS_H
#define BYTEPS_MXNET_MPI_OPS_H

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/c_api_error.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>

namespace byteps {
namespace mxnet {

using namespace byteps::common;

typedef ::mxnet::Engine Engine;
typedef ::mxnet::NDArray NDArray;
typedef ::mxnet::Engine::CallbackOnComplete Callback;

extern "C" int byteps_mxnet_push_async(NDArray* input,
                                       char* name, int priority);

extern "C" int byteps_mxnet_pull_async(NDArray* output,
                                       char* name, int priority);

} // namespace mxnet
} // namespace byteps

#endif // BYTEPS_MXNET_MPI_OPS_H