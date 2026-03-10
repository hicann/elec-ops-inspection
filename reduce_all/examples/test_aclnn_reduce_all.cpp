/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_all.h"

// 测试用例结构
struct TestCase {
    std::string name;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> reduceDims;
    std::vector<int64_t> expectedOutputShape;
    bool keepDim;
};

// ==================== 测试配置 ====================
// 每个测试用例的随机测试次数
const int RANDOM_TEST_ROUNDS = 5;

// 随机数据中 true 的比例（0.0-1.0）
const double TRUE_RATIO = 0.8;

// 随机种子（0表示使用时间种子）
const unsigned int RANDOM_SEED = 0;

// 选择要运行的测试用例索引，-1 表示运行所有
const int RUN_TEST_INDEX = -1;

// 测试用例配置
std::vector<TestCase> testCases = {
    // 基础测试
    {"1D_Simple", {100}, {0}, {1}, false},
    {"1D_Large", {1000000}, {0}, {1}, false},
    
    // // 2D 测试
    {"2D_AllAxis", {4, 3000000}, {0, 1}, {1}, false},
    {"2D_LastAxis", {4, 3000000}, {1}, {4}, false},
    // {"2D_LastAxis", {14, 30000}, {0}, {30000}, false},
    // {"2D_LastAxis", {3000000, 4}, {1}, {3000000}, false},
    {"2D_FirstAxis", {1000, 500}, {0}, {500}, false},
    
    // 3D 测试
    {"3D_MiddleAxis", {4, 30000, 5}, {1}, {4, 5}, false},
    {"3D_MiddleAxis", {400, 300, 500}, {0, 2}, {300}, false},
    {"3D_LastAxis", {4, 3, 5}, {2}, {4, 3}, false},
    {"3D_FirstAxis", {100, 200, 50}, {0}, {200, 50}, false},
    {"3D_MultiAxis", {10, 20, 30}, {0, 2}, {20}, false},
    
    // 4D 测试
    // {"4D_Dim3", {2, 4, 500, 200}, {3}, {2, 4, 500}, false},
    {"4D_Dim23", {2, 4, 500, 200}, {2, 3}, {2, 4}, false},
    // {"4D_Dim1", {2, 4, 5, 2000}, {1}, {2, 5, 2000}, false},
    {"4D_Dim2", {2, 400, 5, 2}, {2}, {2, 400, 2}, false},
    {"4D_Dim12", {2, 4, 5, 200}, {1, 2}, {2, 200}, false},
    {"4D_Dim02", {2, 4, 5, 2}, {0, 2}, {4, 2}, false},
    // {"4D_Dim23_Large", {2, 4, 8, 1600000}, {2, 3}, {2, 4}, false},
    // {"4D_Dim23_Large", {2, 4, 8, 1600000}, {1, 2}, {2, 1600000}, false},
    {"4D_Dim1_Large", {4, 4, 8, 1600}, {1}, {4, 8, 1600}, false},
    // {"4D_Dim1_Large", {200, 4, 8, 1600}, {1}, {200, 8, 1600}, false},
    {"4D_Dim1_Small", {2, 4, 8, 8}, {1}, {2, 8, 8}, false},
    {"4D_Dim1_Small2", {2, 8, 8, 8}, {1}, {2, 8, 8}, false},
    // {"4D_Dim2_Large", {2, 4, 800000, 16}, {2}, {2, 4, 16}, false},
    // {"4D_Dim12_Large", {2, 4, 8, 160000}, {1, 2}, {2, 160000}, false},
    {"4D_Dim02_Med", {2, 4, 800, 16}, {0, 2}, {4, 16}, false},
    {"4D_AllAxis", {2, 3, 4, 5}, {0, 1, 2, 3}, {1}, false},
    
    // keepDim 测试
    // {"4D_KeepDim_Dim1", {2, 4, 5, 200}, {1}, {2, 1, 5, 200}, true},
    {"4D_KeepDim_Dim23", {2, 4, 5, 200}, {2, 3}, {2, 4, 1, 1}, true},
    {"3D_KeepDim_All", {3, 4, 5}, {0, 1, 2}, {1, 1, 1}, true},
    
    // // 负数维度测试
    {"3D_NegDim", {10, 20, 30}, {-1}, {10, 20}, false},
    {"4D_NegDims", {2, 3, 4, 5}, {-1, -3}, {2, 4}, false},
    
    // 边界情况
    {"SingleElement", {1}, {0}, {1}, false},
};
// =====================================================

// 全局随机数生成器
std::mt19937 g_rng;

int64_t calcTotalSize(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

std::vector<int64_t> calcOutputShape(const std::vector<int64_t>& inputShape,
                                      const std::vector<int64_t>& reduceDims,
                                      bool keepDim) {
    std::vector<int64_t> outputShape;
    for (size_t i = 0; i < inputShape.size(); i++) {
        bool isReduceDim = false;
        for (auto dim : reduceDims) {
            int64_t actualDim = dim < 0 ? dim + static_cast<int64_t>(inputShape.size()) : dim;
            if (actualDim == static_cast<int64_t>(i)) {
                isReduceDim = true;
                break;
            }
        }
        if (isReduceDim) {
            if (keepDim) {
                outputShape.push_back(1);
            }
        } else {
            outputShape.push_back(inputShape[i]);
        }
    }
    if (outputShape.empty()) {
        outputShape.push_back(1);
    }
    return outputShape;
}

std::vector<int64_t> calcStrides(const std::vector<int64_t>& shape) {
    if (shape.empty()) return {1};
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

std::string shapeToString(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        result += std::to_string(shape[i]);
        if (i < shape.size() - 1) result += ", ";
    }
    result += "]";
    return result;
}

// 生成随机布尔数据
void generateRandomBoolData(std::vector<uint8_t>& data, double trueRatio) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& val : data) {
        val = (dist(g_rng) < trueRatio) ? 1 : 0;
    }
}

// 生成特定模式的数据用于边界测试
enum class DataPattern {
    RANDOM,      // 随机数据
    ALL_TRUE,    // 全为 true
    ALL_FALSE,   // 全为 false
    SINGLE_FALSE,// 只有一个 false
    SINGLE_TRUE, // 只有一个 true
    ALTERNATING  // 交替
};

void generatePatternData(std::vector<uint8_t>& data, DataPattern pattern) {
    switch (pattern) {
        case DataPattern::ALL_TRUE:
            std::fill(data.begin(), data.end(), 1);
            break;
        case DataPattern::ALL_FALSE:
            std::fill(data.begin(), data.end(), 0);
            break;
        case DataPattern::SINGLE_FALSE:
            std::fill(data.begin(), data.end(), 1);
            if (!data.empty()) {
                std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
                data[dist(g_rng)] = 0;
            }
            break;
        case DataPattern::SINGLE_TRUE:
            std::fill(data.begin(), data.end(), 0);
            if (!data.empty()) {
                std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
                data[dist(g_rng)] = 1;
            }
            break;
        case DataPattern::ALTERNATING:
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = (i % 2 == 0) ? 1 : 0;
            }
            break;
        case DataPattern::RANDOM:
        default:
            generateRandomBoolData(data, TRUE_RATIO);
            break;
    }
}

// CPU 端计算 ReduceAll 的期望结果
void computeExpectedReduceAll(const std::vector<uint8_t>& inputData,
                               const std::vector<int64_t>& inputShape,
                               const std::vector<int64_t>& reduceDims,
                               const std::vector<int64_t>& outputShape,
                               bool keepDim,
                               std::vector<uint8_t>& expectedOutput) {
    int64_t inputSize = calcTotalSize(inputShape);
    int64_t outputSize = calcTotalSize(outputShape);
    std::vector<int64_t> inputStrides = calcStrides(inputShape);
    
    // 初始化输出为全 true
    expectedOutput.assign(outputSize, 1);
    
    // 规范化 reduceDims
    std::vector<int64_t> normalizedReduceDims;
    for (auto dim : reduceDims) {
        int64_t actualDim = dim < 0 ? dim + static_cast<int64_t>(inputShape.size()) : dim;
        normalizedReduceDims.push_back(actualDim);
    }
    std::sort(normalizedReduceDims.begin(), normalizedReduceDims.end());
    
    // 遍历输入的每个元素
    std::vector<int64_t> inputIndices(inputShape.size(), 0);
    
    for (int64_t flatIdx = 0; flatIdx < inputSize; flatIdx++) {
        // 计算当前的多维索引
        int64_t remaining = flatIdx;
        for (size_t d = 0; d < inputShape.size(); d++) {
            inputIndices[d] = remaining / inputStrides[d];
            remaining %= inputStrides[d];
        }
        
        // 计算对应的输出索引
        std::vector<int64_t> outputIndices;
        for (size_t d = 0; d < inputShape.size(); d++) {
            bool isReduceDim = std::find(normalizedReduceDims.begin(), 
                                          normalizedReduceDims.end(), 
                                          static_cast<int64_t>(d)) != normalizedReduceDims.end();
            if (isReduceDim) {
                if (keepDim) {
                    outputIndices.push_back(0);
                }
            } else {
                outputIndices.push_back(inputIndices[d]);
            }
        }
        
        // 处理空输出形状的情况
        if (outputIndices.empty()) {
            outputIndices.push_back(0);
        }
        
        // 计算输出的扁平索引
        std::vector<int64_t> outputStrides = calcStrides(outputShape);
        int64_t outputFlatIdx = 0;
        for (size_t d = 0; d < outputIndices.size(); d++) {
            outputFlatIdx += outputIndices[d] * outputStrides[d];
        }
        
        // ReduceAll: 如果任何输入为 false，对应输出为 false
        if (inputData[flatIdx] == 0) {
            expectedOutput[outputFlatIdx] = 0;
        }
    }
}

// 比较结果
bool compareResults(const std::vector<uint8_t>& actual, 
                    const std::vector<uint8_t>& expected,
                    int maxErrors = 10) {
    if (actual.size() != expected.size()) {
        std::cout << "[错误] 结果大小不匹配: 实际=" << actual.size() 
                  << ", 期望=" << expected.size() << std::endl;
        return false;
    }
    
    int errorCount = 0;
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            if (errorCount < maxErrors) {
                std::cout << "[错误] 位置 " << i << ": 实际=" << (int)actual[i] 
                          << ", 期望=" << (int)expected[i] << std::endl;
            }
            errorCount++;
        }
    }
    
    if (errorCount > 0) {
        std::cout << "[错误] 总共 " << errorCount << " 个位置不匹配" << std::endl;
        return false;
    }
    return true;
}

std::string patternToString(DataPattern pattern) {
    switch (pattern) {
        case DataPattern::RANDOM: return "随机";
        case DataPattern::ALL_TRUE: return "全True";
        case DataPattern::ALL_FALSE: return "全False";
        case DataPattern::SINGLE_FALSE: return "单个False";
        case DataPattern::SINGLE_TRUE: return "单个True";
        case DataPattern::ALTERNATING: return "交替";
        default: return "未知";
    }
}

bool runSingleTest(const TestCase& tc, aclrtStream stream, 
                   DataPattern pattern, int round) {
    std::vector<int64_t> outputShape = calcOutputShape(tc.inputShape, tc.reduceDims, tc.keepDim);
    std::vector<int64_t> inputStrides = calcStrides(tc.inputShape);
    std::vector<int64_t> outputStrides = calcStrides(outputShape);
    
    int64_t inputSize = calcTotalSize(tc.inputShape);
    int64_t outputSize = calcTotalSize(outputShape);
    
    // 准备输入数据
    std::vector<uint8_t> inputData(inputSize);
    generatePatternData(inputData, pattern);
    
    // 计算期望输出
    std::vector<uint8_t> expectedOutput;
    computeExpectedReduceAll(inputData, tc.inputShape, tc.reduceDims, 
                             outputShape, tc.keepDim, expectedOutput);
    
    std::vector<uint8_t> outputData(outputSize, 0);
    
    // 统计输入数据
    int64_t trueCount = std::count(inputData.begin(), inputData.end(), 1);
    int64_t falseCount = inputSize - trueCount;
    
    std::cout << "  [轮次 " << round << "] 模式: " << patternToString(pattern)
              << ", True: " << trueCount << ", False: " << falseCount << std::endl;
    
    // 分配设备内存
    void* inputDevAddr = nullptr;
    void* outputDevAddr = nullptr;
    aclrtMalloc(&inputDevAddr, inputSize * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outputDevAddr, outputSize * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputDevAddr, inputSize, inputData.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    // 创建 Tensor
    aclTensor* inputTensor = aclCreateTensor(
        tc.inputShape.data(), tc.inputShape.size(), ACL_BOOL,
        inputStrides.data(), 0, ACL_FORMAT_ND,
        tc.inputShape.data(), tc.inputShape.size(), inputDevAddr);
    
    aclTensor* outputTensor = aclCreateTensor(
        outputShape.data(), outputShape.size(), ACL_BOOL,
        outputStrides.data(), 0, ACL_FORMAT_ND,
        outputShape.data(), outputShape.size(), outputDevAddr);
    
    aclIntArray* dims = aclCreateIntArray(tc.reduceDims.data(), tc.reduceDims.size());
    
    // 执行 ReduceAll
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    
    auto ret = aclnnReduceAllGetWorkspaceSize(inputTensor, dims, tc.keepDim, outputTensor, &workspaceSize, &executor);
    if (ret != 0) {
        std::cout << "  [错误] aclnnReduceAllGetWorkspaceSize 失败, 错误码: " << ret << std::endl;
        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
        aclDestroyIntArray(dims);
        aclrtFree(inputDevAddr);
        aclrtFree(outputDevAddr);
        return false;
    }
    
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    
    ret = aclnnReduceAll(workspaceAddr, workspaceSize, executor, stream);
    if (ret != 0) {
        std::cout << "  [错误] aclnnReduceAll 失败, 错误码: " << ret << std::endl;
        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
        aclDestroyIntArray(dims);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclrtFree(inputDevAddr);
        aclrtFree(outputDevAddr);
        return false;
    }
    
    aclrtSynchronizeStream(stream);
    
    // 拷回结果
    aclrtMemcpy(outputData.data(), outputSize, outputDevAddr, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    // 验证结果
    bool passed = compareResults(outputData, expectedOutput);
    
    if (!passed) {
        // 打印更多调试信息
        const int64_t printN = std::min<int64_t>(outputSize, 10);
        std::cout << "  实际输出前10个: [";
        for (int64_t i = 0; i < printN; i++) {
            std::cout << (int)outputData[i];
            if (i < printN - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  期望输出前10个: [";
        for (int64_t i = 0; i < printN; i++) {
            std::cout << (int)expectedOutput[i];
            if (i < printN - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "  [" << (passed ? "通过" : "失败") << "]" << std::endl;
    
    // 清理资源
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyIntArray(dims);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtFree(inputDevAddr);
    aclrtFree(outputDevAddr);
    
    return passed;
}

bool runTestCase(const TestCase& tc, aclrtStream stream) {
    std::cout << "\n========== 测试: " << tc.name << " ==========" << std::endl;
    std::cout << "输入形状: " << shapeToString(tc.inputShape) << std::endl;
    std::cout << "规约维度: " << shapeToString(tc.reduceDims) << std::endl;
    std::cout << "保持维度: " << (tc.keepDim ? "是" : "否") << std::endl;
    
    std::vector<int64_t> outputShape = calcOutputShape(tc.inputShape, tc.reduceDims, tc.keepDim);
    int64_t inputSize = calcTotalSize(tc.inputShape);
    int64_t outputSize = calcTotalSize(outputShape);
    
    std::cout << "计算输出形状: " << shapeToString(outputShape) << std::endl;
    std::cout << "期望输出形状: " << shapeToString(tc.expectedOutputShape) << std::endl;
    std::cout << "输入元素数: " << inputSize << std::endl;
    std::cout << "输出元素数: " << outputSize << std::endl;
    
    // 验证输出形状
    if (outputShape != tc.expectedOutputShape) {
        std::cout << "[错误] 输出形状不匹配!" << std::endl;
        return false;
    }
    
    int passCount = 0;
    int totalTests = 0;
    
    // 首先运行边界模式测试
    std::vector<DataPattern> boundaryPatterns = {
        DataPattern::ALL_TRUE,
        DataPattern::ALL_FALSE,
        DataPattern::SINGLE_FALSE,
        DataPattern::SINGLE_TRUE
    };
    
    std::cout << "\n--- 边界模式测试 ---" << std::endl;
    for (auto pattern : boundaryPatterns) {
        totalTests++;
        if (runSingleTest(tc, stream, pattern, totalTests)) {
            passCount++;
        }
    }
    
    // 然后运行随机测试
    std::cout << "\n--- 随机数据测试 ---" << std::endl;
    for (int round = 0; round < RANDOM_TEST_ROUNDS; round++) {
        totalTests++;
        if (runSingleTest(tc, stream, DataPattern::RANDOM, totalTests)) {
            passCount++;
        }
    }
    
    bool allPassed = (passCount == totalTests);
    std::cout << "\n[测试 " << tc.name << " 汇总] 通过: " << passCount 
              << "/" << totalTests << " - " << (allPassed ? "全部通过" : "存在失败") << std::endl;
    
    return allPassed;
}

int main()
{
    // 初始化随机数生成器
    unsigned int seed = RANDOM_SEED;
    if (seed == 0) {
        seed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
    g_rng.seed(seed);
    std::cout << "随机种子: " << seed << std::endl;
    
    // 初始化 ACL
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    int passCount = 0;
    int failCount = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (RUN_TEST_INDEX >= 0 && RUN_TEST_INDEX < static_cast<int>(testCases.size())) {
        // 运行单个测试
        if (runTestCase(testCases[RUN_TEST_INDEX], stream)) {
            passCount++;
        } else {
            failCount++;
        }
    } else {
        // 运行所有测试
        for (size_t i = 0; i < testCases.size(); i++) {
            std::cout << "\n===============================================";
            std::cout << "\n[测试用例 " << i + 1 << "/" << testCases.size() << "]";
            if (runTestCase(testCases[i], stream)) {
                passCount++;
            } else {
                failCount++;
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // 打印统计
    std::cout << "\n================================================" << std::endl;
    std::cout << "============= 最终测试统计 =============" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "测试用例通过: " << passCount << std::endl;
    std::cout << "测试用例失败: " << failCount << std::endl;
    std::cout << "测试用例总计: " << (passCount + failCount) << std::endl;
    std::cout << "每用例随机轮次: " << RANDOM_TEST_ROUNDS << std::endl;
    std::cout << "随机数据True比例: " << (TRUE_RATIO * 100) << "%" << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "随机种子: " << seed << std::endl;
    std::cout << "================================================" << std::endl;
    
    // 清理
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return failCount > 0 ? 1 : 0;
}