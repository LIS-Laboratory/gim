
#pragma once

#include <vector>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "kernels.h"
#include "./cub/cub/cub.cuh"

#define MAX_NUMBER_OF_RR_SETS 30000000

class Math{
    public:
        static double log2(int n){
            return log(n) / log(2);
        }
        static double logcnk(int n, int k)
        {
            double ans = 0;
            for(int i = n - k + 1; i <= n; i++){
                ans += log(i);
            }
            for(int i = 1; i <= k; i++){
                ans -= log(i);
            }
            return ans;
        }
};

class IM
{
    private:
        bool IC;
        double epsilon;
        int n_nodes;
        uint32_t seed = 65465;
        const int N_RR = MAX_NUMBER_OF_RR_SETS;
        int N_BLOCKS;// = 80 * 16;
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> dests;
        std::vector<float> probs;

        thrust::device_vector<uint32_t> d_dests;
        thrust::device_vector<float> d_probs;
        thrust::device_vector<uint32_t> d_offsets;
        thrust::device_vector<uint32_t> d_rr_atomic_offset;
        thrust::device_vector<uint32_t> d_rr_sets;
        thrust::device_vector<int> d_heap_overflow;
        
        thrust::device_vector<uint32_t> d_rr_offsets0;
        thrust::device_vector<uint32_t> d_rr_offsets1;
        thrust::device_vector<uint32_t> d_n_rr_sets;
        thrust::device_vector<int> d_visited_cnt;
        thrust::device_vector<int> d_visited_cnt_back;
        thrust::device_vector<int> d_covered_flags;
        thrust::device_vector<cub::KeyValuePair<int, int>> d_argmax;


        void* d_temp_reduce_storage = NULL;
        size_t   temp_reduce_storage_bytes = 0;
        cub::DoubleBuffer<uint32_t> d_rr_offsets_db;
        void* d_temp_sort_storage = NULL;
        size_t temp_sort_storage_bytes = 0;

        uint32_t *d_visited_flags;
        size_t visited_flags_pitch;

        thrust::device_vector<BufferChunk> d_block_buffer;
        size_t block_buffer_pitch;

        void create_csr(const char* filename, const bool reversed);
        double estimate_opt(int budget);
        uint32_t sample_rr_sets(uint32_t n_req);
        double greedy_seed_selection(uint32_t nrr, int budget, bool last=false, std::vector<int>* seed_set=nullptr);


    public:
        IM(const char* filename, const double epsilon, const bool reversed=true);
        ~IM();
        double maximizeInfluence(int budget, std::vector<int>& seed_set, bool IC=true);
};