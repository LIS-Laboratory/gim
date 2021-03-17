
#include "IM.h"

#include <fstream>
#include <numeric>


IM::~IM()
{
    HANDLE_ERROR(cudaFree(d_temp_reduce_storage));
    HANDLE_ERROR(cudaFree(d_temp_sort_storage));
    HANDLE_ERROR(cudaFree(d_visited_flags));
}

IM::IM(const char* filename, const double epsilon, const bool reversed) : epsilon{epsilon}
{
    create_csr(filename, reversed);
    n_nodes = offsets.size() - 1;

    std::cout << "# of nodes: " << n_nodes << std::endl;
    std::cout << "# of edges: " << dests.size() << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    N_BLOCKS = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor/prop.warpSize/2);

    size_t heap_size = 1024ULL*1024UL*128UL;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

    d_dests = dests;
    d_probs = probs;
    d_offsets = offsets;
    d_rr_atomic_offset = thrust::device_vector<uint32_t>(1, 0);
    d_heap_overflow = thrust::device_vector<int>(1, 0);
    d_rr_offsets0 = thrust::device_vector<uint32_t>((uint32_t)(1.1 * N_RR), 0);
    d_rr_offsets1 = thrust::device_vector<uint32_t>((uint32_t)(1.1 * N_RR), 0);
    d_n_rr_sets = thrust::device_vector<uint32_t>(1, 0);
    d_visited_cnt = thrust::device_vector<int>(offsets.size(), 0);
    d_visited_cnt_back = thrust::device_vector<int>(offsets.size(), 0);
    d_covered_flags = thrust::device_vector<int>((uint32_t)(1.1 * N_RR), 0);
    d_argmax = thrust::device_vector<cub::KeyValuePair<int, int>>(1);

    d_temp_reduce_storage = NULL;
    // check how much memory is required for reduction
    cub::DeviceReduce::ArgMax(d_temp_reduce_storage, temp_reduce_storage_bytes, d_visited_cnt.begin(), d_argmax.begin(), offsets.size());
    // allocate the required memory for recudtion
    cudaMalloc(&d_temp_reduce_storage, temp_reduce_storage_bytes);

    d_rr_offsets_db = cub::DoubleBuffer<uint32_t>(thrust::raw_pointer_cast(d_rr_offsets0.data()), thrust::raw_pointer_cast(d_rr_offsets1.data()));

    d_temp_sort_storage = NULL;
    // check how much auxiliary memory is required for sorting
    cub::DeviceRadixSort::SortKeys(d_temp_sort_storage, temp_sort_storage_bytes, d_rr_offsets_db, d_rr_offsets0.size());
    // allocate the required momory for soring
    cudaMalloc(&d_temp_sort_storage, temp_sort_storage_bytes);

    HANDLE_ERROR(cudaMallocPitch((void**)&d_visited_flags, &visited_flags_pitch, (n_nodes+31)/32  * sizeof(uint32_t), N_BLOCKS));
    d_block_buffer = thrust::device_vector<BufferChunk>(N_BLOCKS);

    size_t l_free = 0;
    size_t l_Total = 0;
    // check how much free memory is remaining
    HANDLE_ERROR(cudaMemGetInfo(&l_free, &l_Total));
    // std::cout << "allocated " << (l_free/sizeof(uint32_t)/10)*9 << std::endl;
    // allocate 90% of the remaining free memory for random RR sets
    d_rr_sets = thrust::device_vector<uint32_t>((l_free/sizeof(uint32_t)/10)*9, 0);
}


double IM::greedy_seed_selection(uint32_t nrr, int budget, bool last, std::vector<int>* seed_set)
{
    thrust::device_vector<int>& d_visited_cnt_ref = last? d_visited_cnt : d_visited_cnt_back;
    if( !last ){
        thrust::copy_n(thrust::device, d_visited_cnt.begin(), d_visited_cnt.size(), d_visited_cnt_back.begin());
    }

    if( seed_set != nullptr ){
        seed_set->clear();
    }

    thrust::fill_n(thrust::device, d_covered_flags.begin(), d_covered_flags.size(), 0);
    double coverage = 0.0;

    for(int k = 0; k < budget; k++){
        
        // find the node which covers most of unflagged RR sets
        cub::DeviceReduce::ArgMax(d_temp_reduce_storage, temp_reduce_storage_bytes, d_visited_cnt_ref.begin(), d_argmax.begin(), offsets.size());
        
        cub::KeyValuePair<int, int> selected_node = d_argmax[0];
        coverage += selected_node.value;
        if(seed_set != nullptr){
            seed_set->push_back(selected_node.key);
        }

        // flag all RR sets that include the maximum covering node
        if(k != budget-1){
            cover_rr_sets<<<N_BLOCKS, 32>>>(
                selected_node.key,
                thrust::raw_pointer_cast(d_covered_flags.data()),
                thrust::raw_pointer_cast(d_visited_cnt_ref.data()),
                nrr,
                thrust::raw_pointer_cast(d_rr_sets.data()),
                thrust::raw_pointer_cast(d_n_rr_sets.data()),
                d_rr_offsets_db.Current()
            );
            cudaDeviceSynchronize();
        }
    }
    return coverage / nrr;
}

uint32_t IM::sample_rr_sets(uint32_t n_req)
{
    seed += 2;
    if(IC){
        create_rr_sets<<<N_BLOCKS, 32>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            d_visited_flags, visited_flags_pitch,
            // d_block_buffer, block_buffer_pitch,
            thrust::raw_pointer_cast(d_block_buffer.data()),
            seed, n_nodes,
            n_req,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            // thrust::raw_pointer_cast(d_n_alloc.data()),
            thrust::raw_pointer_cast(d_rr_atomic_offset.data()),
            d_rr_offsets_db.Current(),
            d_rr_sets.size(),
            thrust::raw_pointer_cast(d_heap_overflow.data())
            );
    }
    else{
        create_rr_sets_lt<<<N_BLOCKS, 32>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            d_visited_flags, visited_flags_pitch,
            // d_block_buffer, block_buffer_pitch,
            thrust::raw_pointer_cast(d_block_buffer.data()),
            seed, n_nodes,
            n_req,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            // thrust::raw_pointer_cast(d_n_alloc.data()),
            thrust::raw_pointer_cast(d_rr_atomic_offset.data()),
            d_rr_offsets_db.Current(),
            d_rr_sets.size(),
            thrust::raw_pointer_cast(d_heap_overflow.data())
            );
    }
    cudaDeviceSynchronize();
    if(d_rr_atomic_offset[0] > d_rr_sets.size()){
        std::cout << "Global memory is not large enough to store all required RR sets!" << std::endl;
        exit(0);
    }
    const int n_generated_rr = d_n_rr_sets[0];
    cub::DeviceRadixSort::SortKeys(d_temp_sort_storage, temp_sort_storage_bytes, d_rr_offsets_db, n_generated_rr+1);
    return n_generated_rr;
}

double IM::estimate_opt(int budget)
{
    double epsilon_prime = epsilon * sqrt(2);
    std::cout << "Estimating OPT" << std::endl;
    for(int x = 1; ; x++){
        uint32_t ci = (2.0+(2.0/3.0) * epsilon_prime) * ( log(n_nodes) + Math::logcnk(n_nodes, budget) + log(Math::log2(n_nodes))) * pow(2.0, x) / (epsilon_prime* epsilon_prime);

        std::cout << "x: " << x << " ,ci: " << ci << std::endl;
        // std::cout << "ci" << ci << std::endl;
        // std::cout << "offs: " << d_rr_atomic_offset[0] << std::endl;
        if(ci > MAX_NUMBER_OF_RR_SETS){
            std::cout << "Required number of RR sets exceeded MAX_NUMBER_OF_RR_SETS" << std::endl;
            exit(0);
        }
        const int n_generated_rr = sample_rr_sets(ci);
        // std::cout << "genet" << std::endl;
        if(d_heap_overflow[0]){
            std::cout << "Heap size on GPU is too small!" << std::endl;
            exit(0);
        }
        double ept = greedy_seed_selection(n_generated_rr, budget);
        // std::cout << "reed" << std::endl;
        
        if (ept > 1 / pow(2.0, x)){
            double OPT_prime = ept * n_nodes / (1+epsilon_prime);
            std::cout << "Estimating OPT: done!" << std::endl;
            return OPT_prime;
        }
    }
}


double IM::maximizeInfluence(int budget, std::vector<int>& seed_set, bool IC)
{
    double opt_prime;
    this->IC = IC;
    opt_prime = estimate_opt(budget);
    // std::cout << opt_prime << std::endl;

    double e = exp(1);
    double alpha = sqrt(log(n_nodes) + log(2));
    double beta = sqrt((1-1/e) * (Math::logcnk(n_nodes, budget) + log(n_nodes) + log(2)));
    uint32_t R = 2.0 * n_nodes *  pow((1-1/e) * alpha + beta, 2) /  (opt_prime * epsilon * epsilon);

    std::cout << "# of required RR sets: " << R << std::endl;
    std::cout << "Generating RR sets: ";
    uint32_t n_generated_rr = sample_rr_sets(R);
    std::cout << "done!" << std::endl;
    // std::cout << "gen: " << n_generated_rr << std::endl;
    // std::cout << "alloc: " << d_n_alloc[0] << " " << d_n_alloc[1] << std::endl;
    std::cout << "Greedy seed set selection: ";
    double influence = greedy_seed_selection(n_generated_rr, budget, true, &seed_set) * n_nodes;
    std::cout << "done!" << std::endl;
    return influence;
}



void IM::create_csr(const char* filename, const bool reversed)
{
    offsets.resize(1);
    offsets[0] = 0;
    int srcnode, dstnode;
    float prob;
    std::cout << "Reading the input file and creating CSR representation of G: " << std::flush;

    {
        std::ifstream infile(filename);
        while( infile >> srcnode ){
            infile >> dstnode;
            infile >> prob;
            if( srcnode == dstnode ) continue;
            if( reversed ) {std::swap(srcnode, dstnode);}
            if( offsets.size() <= srcnode+1 ){
                offsets.resize(srcnode+2, 0);
            }
            if( offsets.size() <= dstnode+1 ){
                offsets.resize(dstnode+2, 0);
            }
            offsets[srcnode+1]++;
        }
    }

    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    dests.resize(offsets.back());
    probs.resize(offsets.back());

    {
        std::ifstream infile(filename);
        std::vector<uint32_t> place_to_write = offsets;
        while( infile >> srcnode ){
            infile >> dstnode;
            infile >> prob;
            if( srcnode == dstnode ) continue;
            if( reversed ) {std::swap(srcnode, dstnode);}
            probs[ place_to_write[srcnode] ] = prob;
            dests[ place_to_write[srcnode] ] = dstnode;
            place_to_write[srcnode]++;
        }
    }
    std::cout << "done!" << std::endl;
}