
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "kernels.h"
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>
#include "./cub/cub/cub.cuh"

__device__ uint32_t mix_hash(uint32_t a, uint32_t b, uint32_t c)
{
	a=a-b;  a=a-c;  a=a^(c >> 13);
	b=b-c;  b=b-a;  b=b^(a << 8);
	c=c-a;  c=c-b;  c=c^(b >> 13);
	a=a-b;  a=a-c;  a=a^(c >> 12);
	b=b-c;  b=b-a;  b=b^(a << 16);
	c=c-a;  c=c-b;  c=c^(b >> 5);
	a=a-b;  a=a-c;  a=a^(c >> 3);
	b=b-c;  b=b-a;  b=b^(a << 10);
	c=c-a;  c=c-b;  c=c^(b >> 15);
	return c;
}

__device__ void enqueue(uint32_t* q, const uint32_t item, uint32_t* q_tail, const uint32_t q_capacity)
{
	// note that this function doesn't check if queue is full
	uint32_t prev_tail = atomicAdd(q_tail, 1);
    q[prev_tail % q_capacity] = item; 
}

// __device__ uint32_t dequeue(uint32_t* q, uint32_t* q_head, const uint32_t q_capacity)
// {
// 	// note that this function doesn't check if queue is empty
// 	uint32_t prev_head = atomicAdd(q_head, 1);
// 	return q[prev_head % q_capacity];
// }

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
								const uint32_t max_rr_offset, int* heap_overflow)
{
	__shared__ uint32_t sh_q[q_capacity];
	__shared__ uint32_t q_head, q_tail;
	__shared__ int sh_buffer_cnt;
	__shared__ uint32_t sh_n_eob_packages;
	__shared__ int sh_heap_overflow;

	__shared__ QueuePackage* sh_q_package;
	QueuePackage* q_package;

	curandState local_state;

	if(threadIdx.x == 0){
		q_tail = 0;
		sh_n_eob_packages = 0;
		sh_heap_overflow = 0;
	}
	__syncwarp();

	uint32_t rr_cnt = 0;
	const uint32_t tid = blockIdx.x * blockDim.x +threadIdx.x;
	curand_init(seed, tid, 0, &local_state);
	
	visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x*visited_flags_pitch);
	// block_buffer = (int*)((char*)block_buffer + blockIdx.x*block_buffer_pitch);
	BufferChunk* local_block_buffer = &block_buffer[blockIdx.x];

	// int* end_of_block = block_buffer + q_package_size * (n_nodes / q_package_size - 1);

	// clear all visited flags
	for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
		visited_flags[i] = 0;
	}

	// todo: determine the exit condition -> is this already done?
	while(q_tail < n_req_rr_sets){
		rr_cnt++;
		uint32_t cur_node = mix_hash(seed, blockIdx.x, rr_cnt) % n_nodes;
		
		if(threadIdx.x == 0){
			visited_flags[cur_node/32] = 1 << (cur_node%32); //rr_cnt;
			q_head = 0;
			sh_q[0] = cur_node;
			q_tail = 1;
			sh_buffer_cnt = -1;
			local_block_buffer = &block_buffer[blockIdx.x];
			local_block_buffer->next = NULL;
		}
		__syncwarp(); // is this necessary?

		while(q_head != q_tail){
			cur_node = sh_q[q_head % q_capacity];
			__syncwarp();
			if(threadIdx.x == 0){
				q_head++;
				if(sh_buffer_cnt >= 0){
					if(sh_buffer_cnt % BUFFER_CHUCKSIZE == 0 && sh_buffer_cnt != 0){
						BufferChunk* temp = (BufferChunk*)malloc(sizeof(BufferChunk));
						if(!temp){
							// printf("NULL temp\n");
							*heap_overflow = 1;
							sh_heap_overflow = 1;
						}
						else{
							//  atomicAdd(&n_alloc[0], 1);
							temp->next = NULL;
							local_block_buffer->next = temp;
							local_block_buffer = temp;
						}
					}
					local_block_buffer->buffer[sh_buffer_cnt % BUFFER_CHUCKSIZE] = cur_node;
					atomicAdd(&visited_cnt[cur_node], 1);
				}
				sh_buffer_cnt++;
			}
			__syncwarp();
			if(sh_heap_overflow) return;

			uint32_t offset = offsets[cur_node];
			uint32_t n_adjacent = offsets[cur_node + 1] - offset;
			uint32_t n_iteration = (n_adjacent + blockDim.x - 1) / blockDim.x * blockDim.x;

			for(uint32_t i = threadIdx.x; i < n_iteration; i += blockDim.x){
				bool active_iter = i < n_adjacent;
				float edge_prob;
				if(active_iter){
					edge_prob = probs[offset + i];
				}
				else{
					edge_prob = 0.0f; // todo: assign a reasonable value
				}
				 
				if( (active_iter == true) && (curand_uniform(&local_state) < edge_prob) ){
					uint32_t dst = dests[offset + i];
					if( (visited_flags[dst/32] & (1 << (dst%32))) == 0 ){
						atomicOr(&visited_flags[dst/32], 1 << (dst%32));
						enqueue(sh_q, dst, &q_tail, q_capacity);
					}
				}

				// check if the number of elements in q has exceeded q_threshold
				__syncwarp();
				if( q_tail - q_head > q_threshold ){
					if(threadIdx.x == 0){
						QueuePackage* temp = (QueuePackage*)malloc(sizeof(QueuePackage));
						if(!temp){
							// printf("NULL pkg\n");
							*heap_overflow = 1;
						} 
						else{
							// atomicAdd(&n_alloc[1], 1);
							if(sh_n_eob_packages == 0){
								temp->next = NULL;
								temp->prev = NULL;
							}
							else{
								temp->next = NULL;
								temp->prev = q_package;
								q_package->next = temp;
							}
						}
						sh_q_package = temp;
					}
					__syncwarp();
					if(!sh_q_package) return;
					q_package = sh_q_package;

					for(uint32_t i = threadIdx.x; i < q_package_size; i += blockDim.x){
						q_package->package[i] = sh_q[(q_head+i) % q_capacity];
					}
					__syncwarp();
					if(threadIdx.x == 0){
						sh_n_eob_packages++;
						q_head += q_package_size;
					}
					__syncwarp();
				}

				// check if the q is empty, but there is some stored packages
				if( (q_head == q_tail) && (sh_n_eob_packages != 0) ){

					for(uint32_t i = threadIdx.x; i < q_package_size; i += blockDim.x){
						sh_q[(q_tail+i) % q_capacity] = q_package->package[i];
					}

					__syncwarp();
					if(threadIdx.x == 0){
						QueuePackage* temp = q_package->prev;
						if(temp != NULL){
							temp->next = NULL;
						}
						free(q_package);
						// atomicAdd(&n_alloc[1], -1);
						sh_q_package = temp;
						sh_n_eob_packages--;
						q_tail += q_package_size;
						// printf("getting from package\n");
					}
					__syncwarp();
					q_package = sh_q_package;

				}
			}
		}

		// determine the memory location where the generated RR set should be moved to
		__syncwarp();
		if(threadIdx.x == 0){
			q_tail = atomicAdd(n_rr_sets, 1) + 1;
			q_head = atomicAdd(rr_atomic_offset, sh_buffer_cnt);
			// do we need sort rr_offsets? Yes, Indeed!
			rr_offsets[q_tail] = q_head + sh_buffer_cnt;
		}
		__syncwarp();

		// add buffer to rr_sets
		uint32_t offset = q_head;
		uint32_t n_adjacent = sh_buffer_cnt;

		if(offset+n_adjacent > max_rr_offset){
			return;
		}

		local_block_buffer = &block_buffer[blockIdx.x];
		for(uint32_t buf_cnt = 0, j = threadIdx.x; local_block_buffer != NULL; buf_cnt++){
			uint32_t n_cur_buf_items = umin(n_adjacent-buf_cnt*BUFFER_CHUCKSIZE, BUFFER_CHUCKSIZE);
			for(uint32_t i = threadIdx.x; i < n_cur_buf_items ; i += blockDim.x, j+= blockDim.x){
				uint32_t temp = local_block_buffer->buffer[i];
				visited_flags[temp/32] = 0;
				rr_sets[offset + j] = temp;
			}	
			BufferChunk* temp = local_block_buffer->next;
			__syncwarp();
			if(threadIdx.x == 0 && buf_cnt > 0){
				free(local_block_buffer);
				// atomicAdd(&n_alloc[0], -1);
			}
			__syncwarp();
			local_block_buffer = temp;
		}
		

	}
	
}






__global__ void cover_rr_sets(
	int last_node_added, 
	int* covered_flags,
	int* visited_cnt,
	const uint32_t n_req_rr_sets,
	uint32_t* rr_sets,
	uint32_t* n_rr_sets, uint32_t* rr_offsets)
{

	__shared__ bool found_node;

	for(int set_id = blockIdx.x; set_id < n_req_rr_sets; set_id += gridDim.x){

		if( covered_flags[set_id] ) continue;

		uint32_t offset = rr_offsets[set_id];
		uint32_t len = rr_offsets[set_id+1];// - offset;
		if(threadIdx.x == 0){
			if( len < offset)
			printf("aa %i %i\n", offset, len);
		}
		len -= offset;
		if(threadIdx.x == 0){
			found_node = false;
		}
		__syncthreads();


		int n_iteration = (len + blockDim.x - 1) / blockDim.x * blockDim.x;
		for(int i = threadIdx.x; i < n_iteration; i += blockDim.x){
			if(i < len){
				uint32_t node = rr_sets[offset + i];
				if( node == last_node_added ){
					found_node = true;
				}
			}
			__syncthreads();
			if(found_node){
				break;
			}
		}

		if(found_node){
			for(int i = threadIdx.x; i < len; i += blockDim.x){
				uint32_t node = rr_sets[offset + i];
				atomicAdd(&visited_cnt[node], -1);
			}
			if(threadIdx.x == 0){
				covered_flags[set_id] = 1;
			}
		}
		__syncthreads();

	}
}




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
								const uint32_t max_rr_offset, int* heap_overflow)
{
	__shared__ int sh_buffer_cnt;
	__shared__ uint32_t sh_found_edge;
	__shared__ uint32_t sh_cur_node;
	__shared__ int sh_heap_overflow;

	__shared__ typename cub::WarpScan<float>::TempStorage scanner_storage;
	typedef cub::WarpScan<float> WarpScanner;
	curandState local_state;

	if( threadIdx.x == 0 ){
		sh_found_edge = 0;
		sh_heap_overflow = 0;
	}
	__syncwarp();

	uint32_t rr_cnt = 0;

	const uint32_t tib = blockIdx.x * blockDim.x;
	curand_init(seed, tib, 0, &local_state);

	visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x*visited_flags_pitch);
	BufferChunk* local_block_buffer = &block_buffer[blockIdx.x];
	

	// clear all visited flags
	for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
		visited_flags[i] = 0;
	}


	// produce RR set until the total number of RR sets exceeds n_req_rr_sets
	while( sh_found_edge  < n_req_rr_sets){
		rr_cnt++;
		// all threads within the warp randomly select a node
		// because the seed is same for all threads, they select the same node
		uint32_t cur_node = mix_hash(seed, blockIdx.x, rr_cnt) % n_nodes;

		if(threadIdx.x == 0){
			visited_flags[cur_node/32] = 1 << (cur_node%32); //rr_cnt;
			sh_found_edge = 1;
			// sh_buffer_cnt = -1;
			sh_buffer_cnt = 0;
			sh_cur_node = cur_node;
			local_block_buffer = &block_buffer[blockIdx.x];
			local_block_buffer->next = NULL;
		}
		__syncwarp(); // is this necessary?


		// todo: determine the loop condition
		while(sh_found_edge == 1){
			cur_node = sh_cur_node;
			__syncwarp();
			if(threadIdx.x == 0){
				if(sh_buffer_cnt >= 0){
					if(sh_buffer_cnt % BUFFER_CHUCKSIZE == 0 && sh_buffer_cnt != 0){
						 BufferChunk* temp = (BufferChunk*)malloc(sizeof(BufferChunk));
						 if(!temp){
							// printf("NULL temp\n");
							*heap_overflow = 1;
							sh_heap_overflow = 1;
						 }
						 else{
						//  atomicAdd(&n_alloc[0], 1);
						 temp->next = NULL;
						 local_block_buffer->next = temp;
						 local_block_buffer = temp;
						 }
					}
					local_block_buffer->buffer[sh_buffer_cnt % BUFFER_CHUCKSIZE] = cur_node;
					atomicAdd(&visited_cnt[cur_node], 1);
				}
				sh_found_edge = 0;	// not found
				sh_buffer_cnt++;
			}
			__syncwarp();
			if(sh_heap_overflow) return;

			uint32_t offset = offsets[cur_node];
			uint32_t n_adjacent = offsets[cur_node + 1] - offset;

			uint32_t n_iteration = (n_adjacent + blockDim.x - 1) / blockDim.x * blockDim.x;

			float rnd_num = curand_uniform(&local_state);
			float running_sum = 0.0f;


			for(uint32_t i = threadIdx.x; i < n_iteration; i += blockDim.x){
				bool active_iter = i < n_adjacent;
				float edge_prob;
				float inc_prob, exc_prob;
				if(active_iter){
					edge_prob = probs[offset + i];
				}
				else{
					edge_prob = 0.0f; // todo: assign a reasonable value
				}

				WarpScanner(scanner_storage).Scan(edge_prob, inc_prob, exc_prob, running_sum, cub::Sum());
				// if(blockIdx.x == 333) printf("%d, %d, %d: %f %f %f %f\n", threadIdx.x, n_adjacent, n_iteration, edge_prob, inc_prob, exc_prob, rnd_num);
				__syncwarp();
				
				if( active_iter && (rnd_num < inc_prob) && (rnd_num >= exc_prob) ){ // active edge
					uint32_t dst = dests[offset + i];
					if( (visited_flags[dst/32] & (1 << (dst%32))) == 0 ){
						atomicOr(&visited_flags[dst/32], 1 << (dst%32));
						sh_found_edge = 1;	// found an active edge wich its dst han not been visited yet
						sh_cur_node = dst;
					}
					else{
						sh_found_edge = 2; // found active edge, but dst has already been visited 
					}
				}
				__syncwarp();
				running_sum = WarpScanner(scanner_storage).Broadcast(inc_prob, 31);
				__syncwarp();
				if(sh_found_edge){
					break;
				}

			}
		} // end of producing a single RR set


		if(threadIdx.x == 0){
			sh_found_edge = atomicAdd(n_rr_sets, 1) + 1;
			sh_cur_node = atomicAdd(rr_atomic_offset, sh_buffer_cnt);
			rr_offsets[sh_found_edge] = sh_cur_node + sh_buffer_cnt;
		}
		__syncwarp();

		// add buffer to rr_sets
		uint32_t offset = sh_cur_node;
		uint32_t n_adjacent = sh_buffer_cnt;

		if(offset+n_adjacent > max_rr_offset){
			return;
		}

		local_block_buffer = &block_buffer[blockIdx.x];
		for(uint32_t buf_cnt = 0, j = threadIdx.x; local_block_buffer != NULL; buf_cnt++){
			uint32_t n_cur_buf_items = umin(n_adjacent-buf_cnt*BUFFER_CHUCKSIZE, BUFFER_CHUCKSIZE);
			for(uint32_t i = threadIdx.x; i < n_cur_buf_items ; i += blockDim.x, j+= blockDim.x){
				uint32_t temp = local_block_buffer->buffer[i];
				visited_flags[temp/32] = 0;
				rr_sets[offset + j] = temp;
			}	
			BufferChunk* temp = local_block_buffer->next;
			__syncwarp();
			if(threadIdx.x == 0 && buf_cnt > 0){
				free(local_block_buffer);
				// atomicAdd(&n_alloc[0], -1);
			}
			__syncwarp();
			local_block_buffer = temp;
		}

	}
}