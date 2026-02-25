#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdlib.h>
#include "phoenix.h"

const char *file_path = "/mnt/phxfs/test.data";
static int device_id = 0;
static size_t io_size = 1UL * (1 << 10) * (1 << 10) *(1 << 10) + 64 * (1 << 10) *(1 << 10); // 64KB
static size_t io_size_1 = 1UL * (1 << 10) * (1 << 10) *(1 << 10);
// 对齐内存分配函数
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    #ifdef _POSIX_C_SOURCE
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return NULL;
        }
    #else
        // 如果posix_memalign不可用，使用memalign或valloc
        ptr = memalign(alignment, size);
    #endif
    return ptr;
}

// 对齐内存释放
void aligned_free(void* ptr) {
    free(ptr); // memalign/posix_memalign分配的内存可以用free释放
}

int main() {
    void *gpu_buffer, *target_addr;
    void *gpu_buffer_1, *target_addr_1;
    int ret;
    int file_fd;
    ssize_t result; 

    file_fd = open(file_path, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("open failed");
        return 1;
    }

    ret = phxfs_open(device_id);
    if (ret != 0) {
        printf("phxfs init failed: %d\n", ret);
        close(file_fd);
        return 1;
    }
    cudaMalloc(&gpu_buffer, io_size);
    cudaMalloc(&gpu_buffer_1, io_size_1);
    cudaMemset(gpu_buffer, 0x00, io_size);
    cudaMemset(gpu_buffer_1, 0x00, io_size_1);
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = phxfs_regmem(device_id, gpu_buffer, io_size, &target_addr);
    printf("gpu_buffer:%p,target_addr:%p\n",gpu_buffer,target_addr);
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        cudaFree(gpu_buffer);
        // aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    printf("The next begin\n");
    ret = phxfs_regmem(device_id, gpu_buffer_1, io_size_1 ,&target_addr_1);
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        cudaFree(gpu_buffer);
        // aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    printf("gpu_buffer_1:%p,target_addr_1:%p\n",gpu_buffer_1,target_addr_1);
    return 0;
    // 使用对齐的内存分配（512字节对齐，这是O_DIRECT的典型要求）
    void* cpu_buffer = aligned_malloc(io_size, 512);
    if (!cpu_buffer) {
        perror("aligned_malloc failed for cpu_buffer");
        close(file_fd);
        return 1;
    }
    
    // 用模式数据填充CPU缓冲区（例如递增字节）
    for (size_t i = 0; i < io_size; i++) {
        ((unsigned char*)cpu_buffer)[i] = i % 254 +1;
    }
    
    // 检查偏移和大小是否对齐
    off_t offset = 0;
    if (offset % 512 != 0) {
        printf("Error: offset %ld is not 512-byte aligned\n", offset);
        aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    
    if (io_size % 512 != 0) {
        printf("Error: io_size %zu is not 512-byte aligned\n", io_size);
        aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    
    ssize_t written = pwrite(file_fd, cpu_buffer, io_size, offset);
    if (written != (ssize_t)io_size) {
        perror("pwrite failed");
        close(file_fd);
        aligned_free(cpu_buffer);
        return 1;
    }
    printf("Successfully wrote %zu bytes to file\n", io_size);
    
    
    
    phxfs_fileid_t fid;
    fid.fd = file_fd;
    fid.deviceID = device_id; 
    result = phxfs_read(fid, (char*)gpu_buffer+io_size, io_size/2, io_size, 0);
    if (result != io_size) {
        printf("Read file error: expected %zu, got %zd\n", io_size, result);
        phxfs_deregmem(device_id, gpu_buffer, 3*io_size);
        cudaFree(gpu_buffer);
        aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    
    void* gpu_copy = aligned_malloc(io_size, 512);
    if (!gpu_copy) {
        perror("aligned_malloc failed for gpu_copy");
        phxfs_deregmem(device_id, gpu_buffer, 3*io_size);
        cudaFree(gpu_buffer);
        aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    
    // 从GPU复制数据
    cudaError_t cuda_status = cudaMemcpy(gpu_copy, (char*)gpu_buffer+3*io_size/2, io_size, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
        aligned_free(gpu_copy);
        phxfs_deregmem(device_id, gpu_buffer, 3*io_size);
        cudaFree(gpu_buffer);
        aligned_free(cpu_buffer);
        close(file_fd);
        return 1;
    }
    
    // 比较数据
    int match = io_size;
    for(int i = 0; i < io_size; ++i) {
        if(((char*)gpu_copy)[i] != ((char*)cpu_buffer)[i]) {
            // printf("Mismatch at byte %d: GPU=0x%02x, CPU=0x%02x\n", 
            //        i, 
            //        (unsigned char)((char*)gpu_copy)[i], 
            //        (unsigned char)((char*)cpu_buffer)[i]);
            match--;
            
        }
    }
    printf("GPU copy first 10 bytes (hex): ");
    printf("match:%d\n",match);
for(int i = 0; i < 10 && i < io_size; ++i) {
    printf("%02x ", ((unsigned char*)gpu_copy)[i]);
}
printf("\n");
    if (match) {
        printf("SUCCESS: CPU and GPU buffers match!\n");
    } else {
        printf("FAIL: Data mismatch detected\n");
    }
    
    ret = phxfs_deregmem(device_id, gpu_buffer, 3*io_size);
    if (ret) {
        printf("phxfs deregmem failed: %d\n", ret);
    }

    cudaFree(gpu_buffer);
    phxfs_close(device_id);
    
    if (cpu_buffer) {
        aligned_free(cpu_buffer);
    }
    
    if (gpu_copy) {
        aligned_free(gpu_copy);
    }
    
    close(file_fd);
    
  
}