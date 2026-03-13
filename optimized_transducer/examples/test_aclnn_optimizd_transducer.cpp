// Copyright 2026 Electrical Engineering SIG - CANN Community
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

    do {                                                                   \
        aclError ret = (expr);                                             \
        if (ret != ACL_SUCCESS) {                                          \
            std::cerr << "ACL Error " << ret << " at line " << __LINE__    \
                      << ": " << #expr << std::endl;                       \
            return -1;                                                     \
        }                                                                  \
    } while (0)

#define CHECK_PTR(ptr, msg)                                                \
    do {                                                                   \
        if ((ptr) == nullptr) {                                            \
            std::cerr << "Error: " << msg << " at line " << __LINE__ << std::endl; \
            return -1;                                                     \
        }                                                                  \
    } while (0)

int main(int argc, char* argv[]) {
    int32_t deviceId = 0;
    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }

    std::cout << "========================================" << std::endl;
    std::cout << " Optimized Transducer ACLNN Test       " << std::endl;
    std::cout << "========================================" << std::endl;

    // ============ 1. 初始化ACL ============
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    std::cout << "[INFO] ACL initialized on device " << deviceId << std::endl;

    // ============ 2. 定义张量维度 ============
    // 根据你的算子实际定义调整这些参数
    const int32_t batchSize = 2;
    const int32_t maxT = 10;      // 最大时间步
    const int32_t maxU = 5;       // 最大标签长度+1
    const int32_t vocabSize = 32; // 词汇表大小

    // ============ 3. 计算内存大小 ============
    int64_t totalPositions = batchSize * maxT * maxU;  // 压缩格式: Σ(t,u)
    int64_t logitsElements = totalPositions * vocabSize;
    int64_t logitsSize = logitsElements * sizeof(float);
    int64_t targetsElements = batchSize * (maxU - 1);
    int64_t targetsSize = targetsElements * sizeof(int32_t);
    int64_t lengthsSize = batchSize * sizeof(int32_t);
    int64_t lossSize = batchSize * sizeof(float);

    std::cout << "[INFO] Tensor shapes:" << std::endl;
    std::cout << "       logits:  [" << totalPositions << ", " << vocabSize << "]" << std::endl;
    std::cout << "       targets: [" << batchSize << ", " << (maxU - 1) << "]" << std::endl;

    // ============ 4. 分配Host内存 ============
    void *logitsHost = nullptr, *targetsHost = nullptr;
    void *logitLensHost = nullptr, *targetLensHost = nullptr;
    void *lossHost = nullptr;

    CHECK_ACL(aclrtMallocHost(&logitsHost, logitsSize));
    CHECK_ACL(aclrtMallocHost(&targetsHost, targetsSize));
    CHECK_ACL(aclrtMallocHost(&logitLensHost, lengthsSize));
    CHECK_ACL(aclrtMallocHost(&targetLensHost, lengthsSize));
    CHECK_ACL(aclrtMallocHost(&lossHost, lossSize));

    // ============ 5. 初始化输入数据 ============
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> floatDist(-2.0f, 2.0f);
    std::uniform_int_distribution<int32_t> intDist(1, vocabSize - 1);

    // 初始化logits (随机值模拟log概率)
    float* logitsPtr = static_cast<float*>(logitsHost);
    for (int64_t i = 0; i < logitsElements; ++i) {
        logitsPtr[i] = floatDist(gen);
    }

    // 初始化targets (随机标签，避开blank=0)
    int32_t* targetsPtr = static_cast<int32_t*>(targetsHost);
    for (int64_t i = 0; i < targetsElements; ++i) {
        targetsPtr[i] = intDist(gen);
    }

    // 初始化lengths
    int32_t* logitLensPtr = static_cast<int32_t*>(logitLensHost);
    int32_t* targetLensPtr = static_cast<int32_t*>(targetLensHost);
    for (int32_t b = 0; b < batchSize; ++b) {
        logitLensPtr[b] = maxT;       // 使用最大长度
        targetLensPtr[b] = maxU - 1;  // 使用最大标签长度
    }

    std::cout << "[INFO] Input data initialized" << std::endl;

    // ============ 6. 分配Device内存 ============
    void *logitsDev = nullptr, *targetsDev = nullptr;
    void *logitLensDev = nullptr, *targetLensDev = nullptr;
    void *lossDev = nullptr, *gradsDev = nullptr;

    CHECK_ACL(aclrtMalloc(&logitsDev, logitsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&targetsDev, targetsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&logitLensDev, lengthsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&targetLensDev, lengthsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&lossDev, lossSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gradsDev, logitsSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // ============ 7. 拷贝数据到Device ============
    CHECK_ACL(aclrtMemcpy(logitsDev, logitsSize, logitsHost, logitsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(targetsDev, targetsSize, targetsHost, targetsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(logitLensDev, lengthsSize, logitLensHost, lengthsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(targetLensDev, lengthsSize, targetLensHost, lengthsSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout << "[INFO] Data copied to device" << std::endl;

    // ============ 8. 创建aclTensor ============
    // logits: [Σ(t,u), V] - 2D 压缩格式
    std::vector<int64_t> logitsShape = {totalPositions, vocabSize};
    std::vector<int64_t> logitsStrides = {vocabSize, 1};

    aclTensor* logitsTensor = aclCreateTensor(
        logitsShape.data(), logitsShape.size(),
        ACL_FLOAT, logitsStrides.data(), 0,
        ACL_FORMAT_ND, logitsShape.data(), logitsShape.size(),
        logitsDev);
    CHECK_PTR(logitsTensor, "Failed to create logits tensor");

    // targets: [B, U-1]
    std::vector<int64_t> targetsShape = {batchSize, maxU - 1};
    std::vector<int64_t> targetsStrides = {maxU - 1, 1};

    aclTensor* targetsTensor = aclCreateTensor(
        targetsShape.data(), targetsShape.size(),
        ACL_INT32, targetsStrides.data(), 0,
        ACL_FORMAT_ND, targetsShape.data(), targetsShape.size(),
        targetsDev);
    CHECK_PTR(targetsTensor, "Failed to create targets tensor");

    // logitLengths: [B]
    std::vector<int64_t> lengthsShape = {batchSize};
    std::vector<int64_t> lengthsStrides = {1};

    aclTensor* logitLensTensor = aclCreateTensor(
        lengthsShape.data(), lengthsShape.size(),
        ACL_INT32, lengthsStrides.data(), 0,
        ACL_FORMAT_ND, lengthsShape.data(), lengthsShape.size(),
        logitLensDev);
    CHECK_PTR(logitLensTensor, "Failed to create logitLengths tensor");

    // targetLengths: [B]
    aclTensor* targetLensTensor = aclCreateTensor(
        lengthsShape.data(), lengthsShape.size(),
        ACL_INT32, lengthsStrides.data(), 0,
        ACL_FORMAT_ND, lengthsShape.data(), lengthsShape.size(),
        targetLensDev);
    CHECK_PTR(targetLensTensor, "Failed to create targetLengths tensor");

    // loss: [B]
    std::vector<int64_t> lossShape = {batchSize};
    std::vector<int64_t> lossStrides = {1};

    aclTensor* lossTensor = aclCreateTensor(
        lossShape.data(), lossShape.size(),
        ACL_FLOAT, lossStrides.data(), 0,
        ACL_FORMAT_ND, lossShape.data(), lossShape.size(),
        lossDev);
    CHECK_PTR(lossTensor, "Failed to create loss tensor");

    // grads: [Σ(t,u), V] - 与 logits 相同的 2D 压缩格式
    aclTensor* gradsTensor = aclCreateTensor(
        logitsShape.data(), logitsShape.size(),
        ACL_FLOAT, logitsStrides.data(), 0,
        ACL_FORMAT_ND, logitsShape.data(), logitsShape.size(),
        gradsDev);
    CHECK_PTR(gradsTensor, "Failed to create grads tensor");

    std::cout << "[INFO] Tensors created" << std::endl;

// ============ 9. 调用算子 ============
    int64_t blank = 0;              // blank token id
    double clamp = -1.0;            // clamp值，-1表示不做clamp
    bool fused_log_softmax = true; // 是否融合计算softmax

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一步: 获取workspace大小
    aclError ret = aclnnOptimizedTransducerGetWorkspaceSize(
        logitsTensor,       // logits
        targetsTensor,      // targets
        logitLensTensor,    // logitLengths
        targetLensTensor,   // targetLengths
        blank,              // blank id
        clamp,              // clamp值
        fused_log_softmax,  // fused_log_softmax标志
        lossTensor,         // loss输出
        gradsTensor,        // grads输出
        &workspaceSize,
        &executor);

    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclnnOptimizedTransducerGetWorkspaceSize failed: " << ret << std::endl;
        return -1;
    }

    std::cout << "[INFO] Workspace size: " << workspaceSize << " bytes" << std::endl;

    // 分配workspace
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 第二步: 执行算子
    ret = aclnnOptimizedTransducer(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclnnOptimizedTransducer execution failed: " << ret << std::endl;
        return -1;
    }

    // 同步
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::cout << "[INFO] Operator executed successfully" << std::endl;

    // ============ 10. 获取结果 ============
    CHECK_ACL(aclrtMemcpy(lossHost, lossSize, lossDev, lossSize, ACL_MEMCPY_DEVICE_TO_HOST));

    float* lossResult = static_cast<float*>(lossHost);
    std::cout << "\n========== Results ==========" << std::endl;
    for (int32_t b = 0; b < batchSize; ++b) {
        std::cout << "  Sample " << b << " loss: " << lossResult[b] << std::endl;
    }
    std::cout << "==============================" << std::endl;

    // ============ 11. 清理资源 ============
    aclDestroyTensor(logitsTensor);
    aclDestroyTensor(targetsTensor);
    aclDestroyTensor(logitLensTensor);
    aclDestroyTensor(targetLensTensor);
    aclDestroyTensor(lossTensor);
    aclDestroyTensor(gradsTensor);

    if (workspace) aclrtFree(workspace);
    aclrtFree(logitsDev);
    aclrtFree(targetsDev);
    aclrtFree(logitLensDev);
    aclrtFree(targetLensDev);
    aclrtFree(lossDev);
    aclrtFree(gradsDev);

    aclrtFreeHost(logitsHost);
    aclrtFreeHost(targetsHost);
    aclrtFreeHost(logitLensHost);
    aclrtFreeHost(targetLensHost);
    aclrtFreeHost(lossHost);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "\n[INFO] Test completed successfully!" << std::endl;
    return 0;
}