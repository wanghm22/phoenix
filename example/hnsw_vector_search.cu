#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include<curand_kernel.h>
// #include <cufile.h>
#include "phoenix.h"
// 配置参数 - 调整为更合理的值
#define DIM 1024                    // 向量维度
#define NUM_VECTORS 524288        // 减少到512MB数据: 1048576 * 128 * 4 = 512MB
#define NUM_QUERIES 1000           // 减少查询数量
#define TOP_K 6                   // 返回的最近邻数量
#define MAX_EDGES_PER_NODE 8      // 每个节点的最大边数
#define edge 64             // CUDA块大小
#define layer_num 3
// 错误检查宏
using namespace std;
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

char *filename = "/mnt/nvme/hnsw.data";
int device_id = 0;
vector<unsigned int> layer1(0);
vector<unsigned int> layer2(0);
// 简单的优先级队列元素
typedef struct {
    unsigned int id;
    unsigned int distance;
} PQElement;

typedef struct Node{
    
    unsigned int layer;
    unsigned int *neighbor;
}Node;
Node node[NUM_VECTORS];
// 生成随机向量数据
void generate(){
   unsigned int node_num = 0;
   unsigned int *vec = (unsigned int*)malloc(NUM_VECTORS*sizeof(unsigned int));
   int fd;
   fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
   srand(time(NULL));
   int ret = phxfs_open(device_id);
    if (ret != 0) {
        printf("phxfs init failed: %d\n", ret);
        close(fd);
        return ;
    }
   while(node_num<NUM_VECTORS){
        for(int i=0;i<DIM;++i){
            vec[i] = rand()%edge;
        }
        ssize_t written;
        written = pwrite(fd,vec,DIM*sizeof(unsigned int),node_num*DIM*sizeof(unsigned int));
        void* gpu_bf;
        cudaMemcpy(gpu_bf, vec, NUM_VECTORS*sizeof(unsigned int), cudaMemcpyHostToDevice);
        unsigned int l=0;
        unsigned int random=rand();
        if(random%10==0){l+=1;}
        if(random%100==0){l+=1;}
        node[node_num].neighbor = (unsigned int *)malloc((l+1)*MAX_EDGES_PER_NODE*sizeof(unsigned int));
        node[node_num].layer = l;
        insert(node_num,gpu_bf);
        node_num++;
   }
   phxfs_close(device_id);
}
void insert(unsigned int idx,void* gpu_bf){
   unsigned int layeridx = node[idx].layer;
   unsigned int startidx;
   for(unsigned int i = layeridx;i>=0;--i){
    unsigned int numoflayer;//当前层其他向量数量
    switch(i){
        case 0:
        numoflayer = idx;
        if(numoflayer <= MAX_EDGES_PER_NODE){
            for(unsigned int j = 0;j<numoflayer;++j){
                node[j].neighbor[numoflayer-1] = idx;
                node[idx].neighbor[j] = j;
            }
        }
        else{
            unsigned int mindistance = UINT_MAX;
            if(layeridx == 0){
                startidx = rand()%numoflayer;
           }
            int* pq_num = 0;
            PQElement* pq;
            while(1){//找出top-edges的边
               
                int ret;
                int file_fd;
                ssize_t result; 

    file_fd = open(filename, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("open failed");
        return ;
    }

    // ret = phxfs_open(device_id);
    // if (ret != 0) {
    //     printf("phxfs init failed: %d\n", ret);
    //     close(file_fd);
    //     return ;
    // }
    int io_size = DIM*sizeof(unsigned int);
    void**gpu_buffer,**target_addr;
    for(int j=0;j<MAX_EDGES_PER_NODE;++j){

    
    cudaMalloc(&gpu_buffer[j], io_size);
    
    cudaMemset(gpu_buffer[j], 0x00, io_size);
    
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = phxfs_regmem(device_id, gpu_buffer[j], io_size, &target_addr[j]);
    
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        cudaFree(gpu_buffer[j]);
        // aligned_free(cpu_buffer);
        close(file_fd);
        return ;
    }
    phxfs_fileid_t fid;
    fid.fd = file_fd;
    fid.deviceID = device_id; 
    result = phxfs_read(fid, (char*)gpu_buffer[j], 0, io_size, startidx*MAX_EDGES_PER_NODE*sizeof(unsigned int));
    if (result != io_size) {
        printf("Read file error: expected %zu, got %zd\n", io_size, result);
        phxfs_deregmem(device_id, gpu_buffer[j], io_size);
        cudaFree(gpu_buffer[j]);
        
        close(file_fd);
        return ;
    }
}
    PQElement* compute_result = (PQElement*)malloc(MAX_EDGES_PER_NODE*sizeof(PQElement));
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){compute_result[i].id=node[startidx].neighbor[i];compute_result[i].distance=0;}
    compute<<<8,256>>>((unsigned int **)gpu_buffer,(unsigned int*)gpu_bf,compute_result);
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){pq_insert(pq,pq_num,MAX_EDGES_PER_NODE,compute_result[i]);}
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){
        phxfs_deregmem(device_id,gpu_buffer[i],io_size);
        cudaFree(gpu_buffer[i]);
    }//保留最小逻辑还没有写，另外phxf关闭也没写。
    if(mindistance == pq[0].distance){
        for(int j=0;j<MAX_EDGES_PER_NODE;++j){
            node[idx].neighbor[j]=pq[j].id;
        }
        break;}
    else{mindistance = pq[0].distance;}        
}
        }
        case 1:
        numoflayer = layer1.size();
        if(numoflayer <= MAX_EDGES_PER_NODE){
            for(unsigned int j = 0;j<numoflayer;++j){
                node[j].neighbor[MAX_EDGES_PER_NODE+numoflayer-1] = idx;
                node[idx].neighbor[MAX_EDGES_PER_NODE+j] = j;
            }
        }
        else{
            unsigned int mindistance = UINT_MAX;
            if(layeridx == 1){
                
            startidx = layer1[rand()%numoflayer];}
            int* pq_num = 0;
            PQElement* pq;
            while(1){//找出top-edges的边
               
            int ret;
            int file_fd;
            ssize_t result; 

    file_fd = open(filename, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("open failed");
        return ;
    }

    // ret = phxfs_open(device_id);
    // if (ret != 0) {
    //     printf("phxfs init failed: %d\n", ret);
    //     close(file_fd);
    //     return ;
    // }
    int io_size = DIM*sizeof(unsigned int);
    void**gpu_buffer,**target_addr;
    for(int j=0;j<MAX_EDGES_PER_NODE;++j){

    
    cudaMalloc(&gpu_buffer[j], io_size);
    
    cudaMemset(gpu_buffer[j], 0x00, io_size);
    
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = phxfs_regmem(device_id, gpu_buffer[j], io_size, &target_addr[j]);
    
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        cudaFree(gpu_buffer[j]);
        // aligned_free(cpu_buffer);
        close(file_fd);
        return ;
    }
    phxfs_fileid_t fid;
    fid.fd = file_fd;
    fid.deviceID = device_id; 
    result = phxfs_read(fid, (char*)gpu_buffer[j], 0, io_size, startidx*MAX_EDGES_PER_NODE*sizeof(unsigned int));
    if (result != io_size) {
        printf("Read file error: expected %zu, got %zd\n", io_size, result);
        phxfs_deregmem(device_id, gpu_buffer[j], io_size);
        cudaFree(gpu_buffer[j]);
        
        close(file_fd);
        return ;
    }
}
    PQElement* compute_result = (PQElement*)malloc(MAX_EDGES_PER_NODE*sizeof(PQElement));
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){compute_result[i].id=node[startidx].neighbor[i];compute_result[i].distance=0;}
    compute<<<8,256>>>((unsigned int **)gpu_buffer,(unsigned int*)gpu_bf,compute_result);
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){pq_insert(pq,pq_num,MAX_EDGES_PER_NODE,compute_result[i]);}
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){
        phxfs_deregmem(device_id,gpu_buffer[i],io_size);
        cudaFree(gpu_buffer[i]);
    }//保留最小逻辑还没有写，另外phxf关闭也没写。
    if(mindistance == pq[0].distance){
        for(int j=0;j<MAX_EDGES_PER_NODE;++j){
            node[idx].neighbor[MAX_EDGES_PER_NODE+j]=pq[j].id;
        }
        startidx = pq[0].id;
        break;}
    else{mindistance = pq[0].distance;}        
}
        }
        layer1.push_back(idx);
        case 2:
        numoflayer = layer2.size();
        if(numoflayer <= MAX_EDGES_PER_NODE){
            for(unsigned int j = 0;j<numoflayer;++j){
                node[j].neighbor[2*MAX_EDGES_PER_NODE+numoflayer-1] = idx;
                node[idx].neighbor[2*MAX_EDGES_PER_NODE+j] = j;
            }
        }
        else{
            unsigned int mindistance = UINT_MAX;
            if(layeridx == 2){startidx = layer2[rand()%numoflayer];}
            int* pq_num = 0;
            PQElement* pq;
            while(1){//找出top-edges的边
               
            int ret;
            int file_fd;
            ssize_t result; 

    file_fd = open(filename, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("open failed");
        return ;
    }

    // ret = phxfs_open(device_id);
    // if (ret != 0) {
    //     printf("phxfs init failed: %d\n", ret);
    //     close(file_fd);
    //     return ;
    // }
    int io_size = DIM*sizeof(unsigned int);
    void**gpu_buffer,**target_addr;
    for(int j=0;j<MAX_EDGES_PER_NODE;++j){

    
    cudaMalloc(&gpu_buffer[j], io_size);
    
    cudaMemset(gpu_buffer[j], 0x00, io_size);
    
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = phxfs_regmem(device_id, gpu_buffer[j], io_size, &target_addr[j]);
    
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        cudaFree(gpu_buffer[j]);
        // aligned_free(cpu_buffer);
        close(file_fd);
        return ;
    }
    phxfs_fileid_t fid;
    fid.fd = file_fd;
    fid.deviceID = device_id; 
    result = phxfs_read(fid, (char*)gpu_buffer[j], 0, io_size, startidx*MAX_EDGES_PER_NODE*sizeof(unsigned int));
    if (result != io_size) {
        printf("Read file error: expected %zu, got %zd\n", io_size, result);
        phxfs_deregmem(device_id, gpu_buffer[j], io_size);
        cudaFree(gpu_buffer[j]);
        
        close(file_fd);
        return ;
    }
}
    PQElement* compute_result = (PQElement*)malloc(MAX_EDGES_PER_NODE*sizeof(PQElement));
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){compute_result[i].id=node[startidx].neighbor[i];compute_result[i].distance=0;}
    compute<<<8,256>>>((unsigned int **)gpu_buffer,(unsigned int*)gpu_bf,compute_result);
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){pq_insert(pq,pq_num,MAX_EDGES_PER_NODE,compute_result[i]);}
    for(unsigned int i=0;i<MAX_EDGES_PER_NODE;++i){
        phxfs_deregmem(device_id,gpu_buffer[i],io_size);
        cudaFree(gpu_buffer[i]);
    }//保留最小逻辑还没有写，另外phxf关闭也没写。
    if(mindistance == pq[0].distance){
        for(int j=0;j<MAX_EDGES_PER_NODE;++j){
            node[idx].neighbor[2*MAX_EDGES_PER_NODE+j]=pq[j].id;
        }
        startidx = pq[0].id;
        break;}
    else{mindistance = pq[0].distance;}        
}
        }
        layer2.push_back(idx);
    }
   }
}

__global__ void compute(unsigned int** gpu_buffer,unsigned int* gpu_bf,PQElement*compute_result){
   int id = blockIdx.x;
   unsigned int result = 0;
   for(int i=4*threadIdx.x;i<threadIdx.x+4;++i){
       if(gpu_buffer[id][i]<gpu_bf[i]){result+=(gpu_bf[i]-gpu_buffer[id][i])*(gpu_bf[i]-gpu_buffer[id][i]);}
       else{result+=(gpu_buffer[id][i]-gpu_bf[i])*(gpu_buffer[id][i]-gpu_bf[i]);}
   }
   atomicAdd(&compute_result[id].distance, result);
}

// __global__ void generate_random_vectors_kernel(float* vectors, int num_vectors, int dim, unsigned int seed) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_elements = num_vectors * dim;
    
//     if (idx >= total_elements) return;
    
//     // 简单的伪随机数生成
//     unsigned int tid = idx * 1103515245 + seed;
//     tid = (tid * 1103515245 + 12345) & 0x7fffffff;
//     vectors[idx] = (float)(tid % 10000) / 10000.0f;
// }

// // 计算向量距离（L2距离）- 优化版
// __device__ float compute_distance(const float* a, const float* b, int dim) {
//     float dist = 0.0f;
//     int i = 0;
    
//     // 展开循环以提高性能
//     for (; i + 3 < dim; i += 4) {
//         float diff0 = a[i] - b[i];
//         float diff1 = a[i+1] - b[i+1];
//         float diff2 = a[i+2] - b[i+2];
//         float diff3 = a[i+3] - b[i+3];
//         dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
//     }
    
//     // 处理剩余部分
//     for (; i < dim; i++) {
//         float diff = a[i] - b[i];
//         dist += diff * diff;
//     }
    
//     return sqrtf(dist);
// }

// 插入元素到优先级队列（设备函数）
void pq_insert(PQElement* queue, int* size, int max_size, PQElement elem) {
    if (*size == 0) {
        queue[0] = elem;
        *size = 1;
        return;
    }
    
    // 如果队列未满，直接添加到末尾并排序
    if (*size < max_size) {
        int pos = *size;
        queue[pos] = elem;
        (*size)++;
        
        // 向后冒泡排序
        while (pos > 0 && queue[pos].distance < queue[pos-1].distance) {
            PQElement temp = queue[pos];
            queue[pos] = queue[pos-1];
            queue[pos-1] = temp;
            pos--;
        }
    } 
    // 如果队列已满且新元素比最后一个好，替换并排序
    else if (elem.distance < queue[max_size-1].distance) {
        queue[max_size-1] = elem;
        int pos = max_size - 1;
        
        while (pos > 0 && queue[pos].distance < queue[pos-1].distance) {
            PQElement temp = queue[pos];
            queue[pos] = queue[pos-1];
            queue[pos-1] = temp;
            pos--;
        }
    }
}

void search_test(){
   unsigned int node_num = 0;
   unsigned int *vec = (unsigned int*)malloc(NUM_VECTORS*sizeof(unsigned int));
   int fd;
   fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
   srand(time(NULL));
   int ret = phxfs_open(device_id);
    if (ret != 0) {
        printf("phxfs init failed: %d\n", ret);
        close(fd);
        return ;
    }
   while(node_num<NUM_VECTORS){
        for(int i=0;i<DIM;++i){
            vec[i] = rand()%edge;
        }
        ssize_t written;
        written = pwrite(fd,vec,DIM*sizeof(unsigned int),node_num*DIM*sizeof(unsigned int));
        void* gpu_bf;
        cudaMemcpy(gpu_bf, vec, NUM_VECTORS*sizeof(unsigned int), cudaMemcpyHostToDevice);
        unsigned int l=0;
        unsigned int random=rand();
        if(random%10==0){l+=1;}
        if(random%100==0){l+=1;}
        node[node_num].neighbor = (unsigned int *)malloc((l+1)*MAX_EDGES_PER_NODE*sizeof(unsigned int));
        node[node_num].layer = l;
        insert(node_num,gpu_bf);
        node_num++;
   }
   phxfs_close(device_id);
}
// 简化的图搜索核函数 - FIXED VERSION
__global__ void graph_search_kernel(
    float* vectors,           // 所有向量数据 [num_vectors * dim]
    float* queries,           // 查询向量 [num_queries * dim]
    int* graph_edges,         // 图邻接表 [num_vectors * max_edges_per_node]
    int* edge_counts,         // 每个节点的边数 [num_vectors]
    int* results,             // 搜索结果 [num_queries * top_k]
    int num_vectors,
    int dim,
    int num_queries,
    int top_k,
    int max_edges_per_node
) {
    // 获取查询ID
    int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 基本检查 - 如果越界就静默返回
    if (query_id >= num_queries) return;
    
    // 检查指针是否有效（粗略检查，可能无效）
    if (vectors == NULL || queries == NULL || graph_edges == NULL || 
        edge_counts == NULL || results == NULL) {
        return; // 静默失败
    }
    
    // 使用共享内存存储查询向量
    extern __shared__ float query_cache[];
    
    // 计算查询向量的起始位置
    int query_offset = query_id * dim;
    
    // 边界检查 - 如果越界就使用安全值
    if (query_offset < 0 || query_offset + dim > num_queries * dim) {
        return; // 查询向量越界
    }
    
    float* query = queries + query_offset;
    
    // 将查询向量加载到共享内存（忽略可能的越界）
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        // 尝试加载，如果越界则使用0
        if (i >= 0 && i < dim && query_offset + i < num_queries * dim) {
            query_cache[i] = query[i];
        } else {
            query_cache[i] = 0.0f; // 默认值
        }
    }
    __syncthreads();
    
    // 初始化top-k结果
    int local_results[TOP_K];
    float local_distances[TOP_K];
    
    for (int i = 0; i < top_k; i++) {
        local_results[i] = -1;
        local_distances[i] = 1e30f;  // 很大的初始值
    }
    
    // 生成随机起点
    unsigned int seed = query_id * 1103515245 + 12345;
    int start_node = seed % num_vectors;
    
    // 搜索多个节点
    for (int iter = 0; iter < 5; iter++) {
        // 计算当前节点
        int node = (start_node + iter * 1000);
        
        // 确保节点在有效范围内
        if (num_vectors > 0) {
            node = node % num_vectors;
        } else {
            continue;
        }
        
        // 计算节点向量起始位置
        int node_vec_offset = node * dim;
        
        // 计算当前节点距离（忽略可能的越界）
        float dist = 1e30f; // 默认大距离
        
        // 安全地计算距离
        if (node_vec_offset >= 0 && node_vec_offset + dim <= num_vectors * dim) {
            dist = 0.0f;
            
            // 使用共享内存中的查询向量
            for (int i = 0; i < dim; i++) {
                float vec_val = 0.0f;
                // 安全读取向量值
                if (node_vec_offset + i >= 0 && node_vec_offset + i < num_vectors * dim) {
                    vec_val = vectors[node_vec_offset + i];
                }
                float diff = query_cache[i] - vec_val;
                dist += diff * diff;
            }
            dist = sqrtf(dist);
        }
        
        // 更新top-k结果
        for (int k = 0; k < top_k; k++) {
            if (dist < local_distances[k]) {
                // 移动后面的元素
                for (int m = top_k - 1; m > k; m--) {
                    local_results[m] = local_results[m-1];
                    local_distances[m] = local_distances[m-1];
                }
                // 插入新元素
                local_results[k] = node;
                local_distances[k] = dist;
                break;
            }
        }
        
        // 检查当前节点的邻居
        int edge_count = 0;
        
        // 安全读取边计数
        if (node >= 0 && node < num_vectors) {
            edge_count = edge_counts[node];
            
            // 限制边计数范围
            if (edge_count < 0 || edge_count > max_edges_per_node) {
                edge_count = 0;
            }
        }
        
        // 遍历所有邻居
        for (int e = 0; e < edge_count; e++) {
            // 计算边索引
            int edge_idx = node * max_edges_per_node + e;
            int neighbor = -1;
            
            // 安全读取邻居节点
            if (edge_idx >= 0 && edge_idx < num_vectors * max_edges_per_node) {
                neighbor = graph_edges[edge_idx];
            }
            
            // 如果邻居无效，跳过
            if (neighbor < 0 || neighbor >= num_vectors) {
                continue;
            }
            
            // 计算邻居向量起始位置
            int neighbor_vec_offset = neighbor * dim;
            
            // 计算邻居距离
            float neighbor_dist = 1e30f;
            
            if (neighbor_vec_offset >= 0 && neighbor_vec_offset + dim <= num_vectors * dim) {
                neighbor_dist = 0.0f;
                
                // 计算L2距离
                for (int i = 0; i < dim; i++) {
                    float vec_val = 0.0f;
                    if (neighbor_vec_offset + i >= 0 && neighbor_vec_offset + i < num_vectors * dim) {
                        vec_val = vectors[neighbor_vec_offset + i];
                    }
                    float diff = query_cache[i] - vec_val;
                    neighbor_dist += diff * diff;
                }
                neighbor_dist = sqrtf(neighbor_dist);
            }
            
            // 更新top-k结果
            for (int k = 0; k < top_k; k++) {
                if (neighbor_dist < local_distances[k]) {
                    // 移动后面的元素
                    for (int m = top_k - 1; m > k; m--) {
                        local_results[m] = local_results[m-1];
                        local_distances[m] = local_distances[m-1];
                    }
                    // 插入新元素
                    local_results[k] = neighbor;
                    local_distances[k] = neighbor_dist;
                    break;
                }
            }
        }
    }
    
    // 存储结果到全局内存（忽略可能的越界）
    if (query_id >= 0 && query_id < num_queries) {
        int result_offset = query_id * top_k;
        
        for (int i = 0; i < top_k; i++) {
            if (result_offset + i >= 0 && result_offset + i < num_queries * top_k) {
                results[result_offset + i] = local_results[i];
            }
        }
    }
    
    // 确保所有写操作完成
    __threadfence();
}
// 使用GDS将数据写入NVMe SSD
void write_data_with_gds(float* d_data, size_t data_size, const char* filename) {
    printf("Writing data using normal I/O...\n");
    
    // 打开文件（不使用O_DIRECT标志，使用普通I/O）
    int fd = open(filename, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }
    
    // 设置文件大小
    if (ftruncate(fd, data_size) != 0) {
        perror("Failed to set file size");
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    // 分配CPU缓冲区（使用malloc，不是固定内存）
    float* cpu_buffer = (float*)malloc(data_size);
    if (!cpu_buffer) {
        perror("Failed to allocate CPU buffer");
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    // 将数据从GPU拷贝到CPU
    cudaError_t err = cudaMemcpy(cpu_buffer, d_data, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy from GPU to CPU: %s\n", cudaGetErrorString(err));
        free(cpu_buffer);
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    // 计时写入过程
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // 使用普通write写入数据
    ssize_t written = write(fd, cpu_buffer, data_size);
    if (written != (ssize_t)data_size) {
        printf("Write error: expected %zu, got %zd\n", data_size, written);
        free(cpu_buffer);
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    // 确保数据落盘
    fsync(fd);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float write_time;
    CUDA_CHECK(cudaEventElapsedTime(&write_time, start, stop));
    
    printf("  Written: %ld bytes in %.3f ms\n", data_size, write_time);
    printf("  Bandwidth: %.2f GB/s\n", 
           (data_size / (1024.0 * 1024.0 * 1024.0)) / (write_time / 1000.0));
    
    // 清理
    free(cpu_buffer);
    close(fd);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
// 使用GDS从NVMe SSD读取数据到GPU内存
float* read_data_with_gds(size_t data_size, const char* filename, void* target_addr) {
    printf("Reading data using GDS...\n");
    
    // 检查文件是否存在
    struct stat st;
    if (stat(filename, &st) != 0) {
        fprintf(stderr, "File %s does not exist\n", filename);
        return NULL;
    }
    
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("Failed to open file for reading");
        return NULL;
    }
    int ret = phxfs_open(0);
    if (ret != 0) {
        printf("phxfs init failed: %d\n", ret);
        close(fd);
        return NULL;
    }
    void *gpu_buffer;
    cudaMalloc(&gpu_buffer,data_size);
    cudaMemset(gpu_buffer, 0x00, data_size);
    cudaStreamSynchronize(0);
    ret = phxfs_regmem(0, gpu_buffer, data_size, &target_addr);
    printf("gpu_buffer:%p,target_addr:%p\n",gpu_buffer,target_addr);
    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        cudaFree(gpu_buffer);
        // aligned_free(cpu_buffer);
        close(fd);
        return NULL;
    }
    
    // 分配GPU内存
    
    
    // 计时读取过程
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // 使用GDS直接读取到GPU内存
    phxfs_fileid_t fid;
    fid.fd = fd;
    fid.deviceID = 0; 
    int result = phxfs_read(fid, (char*)gpu_buffer, 0, data_size, 0);
    if (result != data_size) {
        printf("Read file error: expected %zu, got %zd\n", data_size, result);
        phxfs_deregmem(0, gpu_buffer, data_size);
        cudaFree(gpu_buffer);
        
        close(fd);
        return NULL;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float read_time;
    CUDA_CHECK(cudaEventElapsedTime(&read_time, start, stop));
    
    printf("  Read: %ld bytes in %.3f ms\n", data_size, read_time);
    printf("  Bandwidth: %.2f GB/s\n", (data_size / (1024.0 * 1024.0 * 1024.0)) / (read_time / 1000.0));
    
    phxfs_deregmem(0, gpu_buffer, data_size);
    close(fd);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return (float*)gpu_buffer;
}

// 构建简单的图索引
void build_graph_index(int* d_graph_edges, int* d_edge_counts, 
                      int num_vectors, int max_edges_per_node) {
    printf("Building graph index...\n");
    
    // 在主机上构建简单的图（随机连接）
    int* h_graph_edges = (int*)malloc(num_vectors * max_edges_per_node * sizeof(int));
    int* h_edge_counts = (int*)malloc(num_vectors * sizeof(int));
    
    srand(time(NULL));
    
    #pragma omp parallel for
    for (int i = 0; i < num_vectors; i++) {
        h_edge_counts[i] = 8 + rand() % 9;  // 8-16条边
        
        // 随机选择邻居（确保不重复且不是自己）
        for (int j = 0; j < h_edge_counts[i]; j++) {
            int neighbor;
            bool duplicate;
            
            do {
                duplicate = false;
                neighbor = rand() % num_vectors;
                
                // 检查是否是自己或已存在
                if (neighbor == i) duplicate = true;
                for (int k = 0; k < j; k++) {
                    if (h_graph_edges[i * max_edges_per_node + k] == neighbor) {
                        duplicate = true;
                        break;
                    }
                }
            } while (duplicate);
            
            h_graph_edges[i * max_edges_per_node + j] = neighbor;
        }
        
        // 填充剩余位置为-1
        for (int j = h_edge_counts[i]; j < max_edges_per_node; j++) {
            h_graph_edges[i * max_edges_per_node + j] = -1;
        }
    }
    
    // 复制到GPU
    CUDA_CHECK(cudaMemcpy(d_graph_edges, h_graph_edges, 
                         num_vectors * max_edges_per_node * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_counts, h_edge_counts, 
                         num_vectors * sizeof(int), 
                         cudaMemcpyHostToDevice));
    
    free(h_graph_edges);
    free(h_edge_counts);
    
    printf("  Built graph with %d nodes, %d max edges per node\n", 
           num_vectors, max_edges_per_node);
}

// 执行向量检索测试
void run_search_test(float* d_vectors, float* d_queries, 
                     int* d_graph_edges,  int* d_edge_counts,
                     float* t_vectors, float* t_queries,
                     int* t_graph_edges,
                     int num_vectors, int num_queries, int dim, 
                     int top_k, int max_edges_per_node) {
    printf("\n=== Starting Vector Search Test ===\n");
    
    // 分配结果内存
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_queries * top_k * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_results, -1, num_queries * top_k * sizeof(int)));
    
    // 配置核函数参数
    int block_size = BLOCK_SIZE;
    int grid_size = (num_queries + block_size - 1) / block_size;
    size_t shared_mem_size = dim * sizeof(float);  // 存储查询向量
    
    printf("Launch configuration:\n");
    printf("  Grid size: %d\n", grid_size);
    printf("  Block size: %d\n", block_size);
    printf("  Shared memory: %ld bytes\n", shared_mem_size);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_latency = 0.0f;
    int num_repeats = 5;
    
    printf("Running %d queries %d times...\n", num_queries, num_repeats);
    
    for (int repeat = 0; repeat < num_repeats; repeat++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        // 执行搜索核函数
        graph_search_kernel<<<grid_size, block_size, shared_mem_size>>>(
            t_vectors,
            t_queries,
            t_graph_edges,
            d_edge_counts,
            d_results,
            num_vectors,
            dim,
            num_queries,
            top_k,
            max_edges_per_node
        );
        
        // 检查内核错误
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
            exit(EXIT_FAILURE);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float latency_ms;
        CUDA_CHECK(cudaEventElapsedTime(&latency_ms, start, stop));
        total_latency += latency_ms;
        
        printf("  Repeat %d: Latency = %7.3f ms\n", repeat + 1, latency_ms);
    }
    
    // 计算吞吐量
    float avg_latency = total_latency / num_repeats;
    float queries_per_second = num_queries / (avg_latency / 1000.0f);
    float throughput = queries_per_second * top_k;  // 每秒处理的向量数
    
    printf("\n=== Performance Summary ===\n");
    printf("Average Latency:   %10.3f ms\n", avg_latency);
    printf("Throughput:        %10.2f QPS (queries per second)\n", queries_per_second);
    printf("Vector Throughput: %10.2f thousand vectors/sec\n", throughput / 1e3);
    printf("Returning top-%d nearest neighbors\n", top_k);
    
    // 验证前3个查询的结果
    int verify_queries = min(3, num_queries);
    int* h_results = (int*)malloc(verify_queries * top_k * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 
                         verify_queries * top_k * sizeof(int), 
                         cudaMemcpyDeviceToHost));
    
    printf("\nFirst %d queries results (node IDs):\n", verify_queries);
    for (int q = 0; q < verify_queries; q++) {
        printf("  Query %d: ", q);
        for (int k = 0; k < top_k; k++) {
            if (h_results[q * top_k + k] >= 0) {
                printf("%6d ", h_results[q * top_k + k]);
            } else {
                printf("%6s ", "---");
            }
        }
        printf("\n");
    }
    
    free(h_results);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_results));
}

int main() {
    printf("=== Vector Search with GDS and Graph Index ===\n");
    
    // 初始化GDS
    // CUfileError_t status = cuFileDriverOpen();
    // if (status.err != CU_FILE_SUCCESS) {
    //     fprintf(stderr, "Failed to initialize GDS driver: %d\n", status.err);
    //     return 1;
    // }
    // printf("GDS driver initialized successfully\n");
    
    // 计算数据大小
    size_t vector_data_size = NUM_VECTORS * DIM * sizeof(float);
    size_t query_data_size = NUM_QUERIES * DIM * sizeof(float);
    size_t graph_edges_size = NUM_VECTORS * MAX_EDGES_PER_NODE * sizeof(int);
    size_t graph_counts_size = NUM_VECTORS * sizeof(int);
    
    char vector_file[] = "/mnt/nvme/vector_data.bin";
    char query_file[] = "/mnt/nvme/query_data.bin";
    char graph_edges_file[] = "/mnt/nvme/graph_edges.bin";
    char graph_counts_file[] = "/mnt/nvme/graph_counts.bin";
    
    printf("\nConfiguration:\n");
    printf("  Vector dimensions:     %d\n", DIM);
    printf("  Number of vectors:     %d\n", NUM_VECTORS);
    printf("  Vector data size:      %.2f MB\n", vector_data_size / (1024.0 * 1024.0));
    printf("  Number of queries:     %d\n", NUM_QUERIES);
    printf("  Max edges per node:    %d\n", MAX_EDGES_PER_NODE);
    printf("  Top-K results:         %d\n", TOP_K);
    
    // 1. 生成向量数据（直接在GPU上生成）
    printf("\n1. Generating vector data on GPU...\n");
    float* d_vectors;
    CUDA_CHECK(cudaMalloc(&d_vectors, vector_data_size));
    
    int threads = 256;
    int blocks = (NUM_VECTORS * DIM + threads - 1) / threads;
    generate_random_vectors_kernel<<<blocks, threads>>>(d_vectors, NUM_VECTORS, DIM, time(NULL));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 2. 使用GDS写入向量数据到NVMe SSD
    printf("\n2. Writing vector data to NVMe SSD using GDS...\n");
    write_data_with_gds(d_vectors, vector_data_size, vector_file);
    
    // 3. 生成查询数据
    printf("\n3. Generating query data...\n");
    float* d_queries;
    CUDA_CHECK(cudaMalloc(&d_queries, query_data_size));
    generate_random_vectors_kernel<<<blocks, threads>>>(d_queries, NUM_QUERIES, DIM, time(NULL) + 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    write_data_with_gds(d_queries, query_data_size, query_file);
    
    // 4. 构建图索引
    printf("\n4. Building graph index...\n");
    int* d_graph_edges;
    int* d_edge_counts;
    CUDA_CHECK(cudaMalloc(&d_graph_edges, graph_edges_size));
    CUDA_CHECK(cudaMalloc(&d_edge_counts, graph_counts_size));
    
    build_graph_index(d_graph_edges, d_edge_counts, NUM_VECTORS, MAX_EDGES_PER_NODE);
    
    // 保存图索引到文件
    write_data_with_gds((float*)d_graph_edges, graph_edges_size, graph_edges_file);
    
    // 5. 使用GDS重新读取数据（模拟实际工作负载）
    printf("\n5. Reading data back using GDS (simulating cold start)...\n");
    
    cudaEvent_t gds_start, gds_stop;
    CUDA_CHECK(cudaEventCreate(&gds_start));
    CUDA_CHECK(cudaEventCreate(&gds_stop));
    
    CUDA_CHECK(cudaEventRecord(gds_start));
    
    // 释放原来的内存，模拟冷启动
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_graph_edges));
    CUDA_CHECK(cudaFree(d_edge_counts));
    
    // 使用GDS重新读取
    void* target_vectors;
    void* target_queries;
    void* target_graph_edges;
    d_vectors = read_data_with_gds(vector_data_size, vector_file, target_vectors);
    d_queries = read_data_with_gds(query_data_size, query_file, target_queries);
    d_graph_edges = (int*)read_data_with_gds(graph_edges_size, graph_edges_file, target_graph_edges);
    
    // 重新分配边计数（这个数据小，直接生成）
    CUDA_CHECK(cudaMalloc(&d_edge_counts, graph_counts_size));
    build_graph_index(d_graph_edges, d_edge_counts, NUM_VECTORS, MAX_EDGES_PER_NODE);
    
    CUDA_CHECK(cudaEventRecord(gds_stop));
    CUDA_CHECK(cudaEventSynchronize(gds_stop));
    
    float gds_time;
    CUDA_CHECK(cudaEventElapsedTime(&gds_time, gds_start, gds_stop));
    float total_mb = (vector_data_size + query_data_size + graph_edges_size) / (1024.0 * 1024.0);
    printf("Total GDS data transfer time: %.3f ms\n", gds_time);
    printf("Average GDS bandwidth: %.2f GB/s\n", (total_mb / 1024.0) / (gds_time / 1000.0));
    
    // 6. 执行向量检索测试
    printf("\n6. Running vector search tests...\n");
    run_search_test(d_vectors, d_queries, d_graph_edges, d_edge_counts,(float*)target_vectors,(float*)target_queries,(int*)target_graph_edges,
                   NUM_VECTORS, NUM_QUERIES, DIM, TOP_K, MAX_EDGES_PER_NODE);
    
    // 7. 清理资源
    printf("\n7. Cleaning up resources...\n");
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_graph_edges));
    CUDA_CHECK(cudaFree(d_edge_counts));
    
    CUDA_CHECK(cudaEventDestroy(gds_start));
    CUDA_CHECK(cudaEventDestroy(gds_stop));
    
    // 删除临时文件
    unlink(vector_file);
    unlink(query_file);
    unlink(graph_edges_file);
    unlink(graph_counts_file);
    
    // 关闭GDS驱动
    // cuFileDriverClose();
    
    printf("\n=== Test Completed Successfully ===\n");
    
    return 0;
}