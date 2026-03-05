#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include "phoenix.h"

const char *file_path = "/mnt/phxfs/test.data";
static int device_id = 0;
static size_t io_size = 64 * (1 << 10); // 64KB

int main() {
    void *gpu_buffer, *target_addr;
    int* cpu_buffer = (int*)malloc(io_size);
    int ret;
    int file_fd;
    ssize_t result; 

    file_fd = open(file_path, O_CREAT | O_RDWR | O_DIRECT, 0644);


    ret = phxfs_open(device_id);

    if (ret != 0) {
        printf("phxfs init failed: %d\n", ret);
        return 1;
    }

    cudaMalloc(&gpu_buffer, io_size);
    cudaMemset(gpu_buffer, 0x00, io_size);
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = phxfs_regmem(device_id, gpu_buffer, io_size, &target_addr);
    printf("Read begin");
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        return 1;
    }
    phxfs_fileid_t fid;
    fid.fd=file_fd;
    fid.deviceID = 0;
    result = phxfs_read(fid, gpu_buffer, io_size, 0 , 0);
    cudaMemcpy(cpu_buffer,gpu_buffer,io_size,cudaMemcpyDeviceToHost);
    for(int i=0;i<1000;++i){
        printf("%d\n",cpu_buffer[i]);
    }
    printf("read over");
    if (result < 0) {
        perror("Read file error");
        return 1;
    }

    ret = phxfs_deregmem(device_id, gpu_buffer, io_size);

    if (ret) {
        printf("phxfs unregmem failed: %d\n", ret);
        return 1;
    }

    cudaFree(gpu_buffer);

    phxfs_close(device_id);

    close(file_fd);
    return 0;
}