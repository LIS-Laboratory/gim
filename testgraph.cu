
#include <vector>
#include <iostream>

#include "IM.h"
#include "kernels.h"


struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};





int main(int argc, char** argv) {

    if( argc < 5 ){
        std::cout << "argv[1]: file containing edge list" << std::endl;
        std::cout << "argv[2]: budget(k)" << std::endl;
        std::cout << "argv[3]: epsilon" << std::endl;
        std::cout << "argv[4]: IC/LT model" << std::endl;
        return -1;
    }

    const char* FILE_NAME = argv[1];
    const int K = std::atoi(argv[2]);
    const double epsilon = std::atof(argv[3]);
    const int ICLT = std::atoi(argv[4]);

    std::cout << "K = " << K << ", eps =  " << epsilon << ", model = " << (ICLT?"IC":"LT") << std::endl;

    std::vector<int> seed_set;

    IM im(FILE_NAME, epsilon);
    GpuTimer gpu_timer;
    gpu_timer.Start();
    double infl = im.maximizeInfluence(K, seed_set, ICLT);
    gpu_timer.Stop();
    std::cout << "Time: " << gpu_timer.Elapsed() << " ms" << std::endl;
    std::cout << infl << std::endl;
    std::cout << "List of nodes in the seed set:";  
    for(auto &a:seed_set){
        std::cout << " " << a;
    }

    std::cout << std::endl;

}