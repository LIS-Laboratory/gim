
#pragma once

#include <iostream>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


// const uint32_t q_capacity = 256;
// const uint32_t q_package_size = 128;
// const uint32_t q_threshold = 256 - 64;

const uint32_t q_capacity = 256;
const uint32_t q_package_size = 32;
const uint32_t q_threshold = q_capacity - q_package_size;

#define BUFFER_CHUCKSIZE 512



struct QueuePackage{
	int package[q_package_size];
	QueuePackage *next;
	QueuePackage *prev;
};


struct BufferChunk{
	int buffer[BUFFER_CHUCKSIZE];
	BufferChunk *next;
};


__global__ void create_rr_sets(
								uint32_t* offsets, uint32_t* dests, const float* probs,
								uint32_t* visited_flags,const size_t visited_flags_pitch,
								// int* block_buffer,const size_t block_buffer_pitch,
								BufferChunk* block_buffer,
								uint32_t seed, const int n_nodes,
								const uint32_t n_req_rr_sets,
								uint32_t* rr_sets,
								int* visited_cnt,
								uint32_t* n_rr_sets,
								// int* n_alloc,
								uint32_t* rr_atomic_offset, uint32_t* rr_offsets,
								const uint32_t max_rr_offset, int* heap_overflow);


__global__ void create_rr_sets_lt(
								uint32_t* offsets, uint32_t* dests, const float* probs,
								uint32_t* visited_flags,const size_t visited_flags_pitch,
								// int* block_buffer,const size_t block_buffer_pitch,
								BufferChunk* block_buffer,
								uint32_t seed, const int n_nodes,
								const uint32_t n_req_rr_sets,
								uint32_t* rr_sets,
								int* visited_cnt,
								uint32_t* n_rr_sets,
								// int* n_alloc,
								uint32_t* rr_atomic_offset, uint32_t* rr_offsets,
								const uint32_t max_rr_offset, int* heap_overflow
								);


__global__ void cover_rr_sets(
								int last_node_added, 
								int* covered_flags,
								int* visited_cnt,
								const uint32_t n_req_rr_sets,
								uint32_t* rr_sets,
								uint32_t* n_rr_sets, uint32_t* rr_offsets);
	