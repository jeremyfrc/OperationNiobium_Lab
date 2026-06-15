#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upSweep(int* scanArray, int offset, int strideVal, int arrayLength){

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int pos = tid * strideVal;

    unsigned int leftIdx = pos + offset - 1;
    unsigned int rightIdx = pos + strideVal - 1;

    if (leftIdx < arrayLength && rightIdx < arrayLength){
        scanArray[rightIdx] += scanArray[leftIdx];
    }
}


__global__ void downSweep(int* scanArray, int offset, int strideVal, int arrayLength){

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int pos = tid * strideVal;

    unsigned int leftIdx = pos + offset - 1;
    unsigned int rightIdx = pos + strideVal - 1;

    if (leftIdx < arrayLength && rightIdx < arrayLength){
        int temp = scanArray[leftIdx];
        scanArray[leftIdx] = scanArray[rightIdx];
        scanArray[rightIdx] += temp;
    }
}

// ============================================================================
// KERNEL 1: 块内局部 Scan + 块总和提取 (完美对齐版)
// ============================================================================
__global__ void local_block_scan(int* device_data, int* block_sums, int N) {
    // 采用标准连续局部共享内存，彻底避免循环内非线性索引漂移
    __shared__ int s_data[ELEMENTS_PER_BLOCK];

    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    int tid = threadIdx.x;

    int idxA = block_offset + 2 * tid;
    int idxB = block_offset + 2 * tid + 1;

    // 1. 规整加载
    s_data[2 * tid]     = (idxA < N) ? device_data[idxA] : 0;
    s_data[2 * tid + 1] = (idxB < N) ? device_data[idxB] : 0;
    __syncthreads();

    // 2. Up-Sweep (Reduction) 阶段：严格在连续逻辑索引上迭代
    int stride = 1;
    for (int d = THREADS_PER_BLOCK; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
        __syncthreads();
    }

    // 3. 完美提取块总和，树顶雷打不动清零
    if (tid == 0) {
        if (block_sums != NULL) {
            block_sums[blockIdx.x] = s_data[ELEMENTS_PER_BLOCK - 1];
        }
        s_data[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    __syncthreads();

    // 4. Down-Sweep 阶段
    for (int d = 1; d <= THREADS_PER_BLOCK; d <<= 1) {
        stride >>= 1;
        if (tid < d) {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            
            int t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
        __syncthreads();
    }

    // 5. 写回全局显存
    if (idxA < N) device_data[idxA] = s_data[2 * tid];
    if (idxB < N) device_data[idxB] = s_data[2 * tid + 1];
}

// ============================================================================
// KERNEL 2: 全局偏移量加法 (完美对齐版)
// ============================================================================
__global__ void accumulate_block_offsets(int* device_data, int* block_sums, int N) {
    if (blockIdx.x == 0) return; 

    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    int tid = threadIdx.x;
    int offset = block_sums[blockIdx.x];

    int idxA = block_offset + 2 * tid;
    int idxB = block_offset + 2 * tid + 1;

    if (idxA < N) device_data[idxA] += offset;
    if (idxB < N) device_data[idxB] += offset;
}


// ============================================================================
// 递归 Multi-pass 引擎 recursive scan
// ============================================================================
void exclusive_scan_recursive(int* device_data, int length) {
    if (length <= 1) return;

    int numBlocks = (length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    
    int* device_block_sums = NULL;
    if (numBlocks > 1) {
        cudaMalloc((void**)&device_block_sums, numBlocks * sizeof(int));
    }

    local_block_scan<<<numBlocks, THREADS_PER_BLOCK>>>(device_data, device_block_sums, length);

    if (numBlocks > 1) {
        exclusive_scan_recursive(device_block_sums, numBlocks);
        accumulate_block_offsets<<<numBlocks, THREADS_PER_BLOCK>>>(device_data, device_block_sums, length);
        cudaFree(device_block_sums);
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
/*
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    if (input != result) {
        cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    //round N up to the next power of 2
    int roundedLength = nextPow2(N);

    int offset = 1;
    int maxOffset = roundedLength/2;

    for (; offset <= maxOffset; offset *= 2){
        int stride = offset * 2;

        int workItems = roundedLength / stride;
        if (workItems * stride < roundedLength) workItems++;

        workItems = (workItems > 0) ? workItems: 1;

        int blockCount = (workItems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        upSweep<<<blockCount, THREADS_PER_BLOCK>>>(result, offset, stride, roundedLength);

        cudaDeviceSynchronize();
    }

    int identity = 0;
    cudaMemcpy(&result[roundedLength - 1], &identity, sizeof(int), cudaMemcpyHostToDevice);

    offset = roundedLength / 2;
    while (offset >= 1){
        int stride = offset * 2;

        int workItems = roundedLength / stride;
        if (workItems * stride < roundedLength) workItems++;

        workItems = (workItems > 0) ? workItems: 1;

        int blockCount = (workItems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        downSweep<<<blockCount, THREADS_PER_BLOCK>>>(result, offset, stride, roundedLength);

        cudaDeviceSynchronize();

        offset /= 2;
    }

}
*/
void exclusive_scan(int* input, int N, int* result) {
    if (input != result) {
        cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    exclusive_scan_recursive(result, N);
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void mark_adjacent_equals(int* inputArray, int* flagArray, int arrayLength){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < arrayLength - 1){
        flagArray[threadId] = (inputArray[threadId] == inputArray[threadId + 1]) ? 1 : 0;
    } else if (threadId == arrayLength - 1) {
        flagArray[threadId] = 0;
    }
}

// find_repeats --
__global__ void extract_repeat_indices(int* flagArray, int* scanResult, int* outputArray, int arrayLength){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < arrayLength - 1 && flagArray[threadId] == 1){
        outputArray[scanResult[threadId]] = threadId;
    }
}
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    if (length <= 1) return 0;
    const size_t intSize = sizeof(int);
    int* deviceFlags = NULL; int* deviceScan = NULL;
    cudaMalloc((void**)&deviceFlags, length * intSize);
    cudaMalloc((void**)&deviceScan, length * intSize);
    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    mark_adjacent_equals<<<numBlocks, THREADS_PER_BLOCK>>>(device_input, deviceFlags, length);
    exclusive_scan(deviceFlags, length, deviceScan);

    int lastFlagValue = 0; int lastScanValue = 0;
    cudaMemcpy(&lastFlagValue, &deviceFlags[length - 1], intSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastScanValue, &deviceScan[length - 1], intSize, cudaMemcpyDeviceToHost);
    int repeatCount = lastFlagValue + lastScanValue;

    if (repeatCount > 0){
        extract_repeat_indices<<<numBlocks, THREADS_PER_BLOCK>>>(deviceFlags, deviceScan, device_output, length);
    }
    cudaFree(deviceFlags); cudaFree(deviceScan);
    return repeatCount;
}
/*
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    const size_t intSize = sizeof(int);
    int repeatCount = 0;
    int lastFlagValue = 0;
    int lastScanValue = 0;
    
    int* deviceFlags = NULL;
    cudaMalloc((void**)&deviceFlags, length * intSize);
    cudaMemset(deviceFlags, 0, length * intSize);

    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(numBlocks);

    //step1: launch kernel to mark repeating elements
    mark_adjacent_equals<<<gridDim, blockDim>>>(device_input, deviceFlags, length);
    cudaDeviceSynchronize();

    int* deviceScan = NULL;
    cudaMalloc((void**)&deviceScan, length * intSize);

    //step2: perform exclusive scan on flags to determine output positions
    //uses the exclusive_scan function
    exclusive_scan(deviceFlags, length, deviceScan);

    //step3: calculate total number of repeats by reading scan results
    if (length > 0){
        //copy last flag and last scan value from device to host
        cudaMemcpy(&lastFlagValue, &deviceFlags[length-1], intSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastScanValue, &deviceScan[length-1], intSize, cudaMemcpyDeviceToHost);

        repeatCount = lastFlagValue + lastScanValue;
    }

    //step4: only process output array if found repeats
    if (repeatCount > 0){
        extract_repeat_indices<<<gridDim, blockDim>>>(deviceFlags, deviceScan, device_output, length);
        cudaDeviceSynchronize();
    }

    //step5: clean up temporary device memory
    cudaFree(deviceFlags);
    cudaFree(deviceScan);

    return repeatCount;
}
*/

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
