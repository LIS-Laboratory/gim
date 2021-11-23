# gIM
gIM is a GPU-accelerated RIS-based influence maximization (IM) algorithm.


# Installation
### CUDA toolkit
To install CUDA toolkit please use [this link](https://developer.nvidia.com/cuda-downloads).

### Repository Preparation

gIM uses cub library for some parallel algorithms like scan. Before compiling the code, initialize submodules using the following command:
 ```
 git submodule update --init
 ```

### Compilation

Compile the code using the following command:
 ```
 nvcc -std=c++11 -O3  -lcurand  *.cu  -o gIM
 ```
 If you are using CUDA 7.5 and newer versions on Windows, you can remove the -std flag because these versions include the support for C++11 standard by default. Wildcard expansion does not work in cmd.exe, so you need to use the following command for compiling the code:
 ```
 nvcc -O3 -lcurand kernels.cu IM.cu testgraph.cu -o gIM
 ``` 
### Execution
 
Run the code using the following command:
```
./gIM /path/to/dataset  k epsilon  IC/LT
```

Description of the arguments:

* /path/to/dataset : The relative or absolute path to the file containing weighted edge list
* k : A positive integer which determines the number of nodes to be put in the seed set
* epsilon : A positive number by which the (1-1/e-epsilon)-approximation ratio is determined
* IC/LT : enter 1 for IC and 0 for LT model
* Example: ```./gIM ../datasets/epin.inf 50 0.1 1```

Input file format:

* The input file is composed of m lines, each of which represents a directed edge by using 3 parameters as follows: ```source-node-id(int) destination-node-id(int)  weight(float)```
* Node IDs are non-negative integers. 
* There is no restriction on the order of edges or existence of parallel edges in the input file.

# Publication

Soheil Shahrouz, Saber Salehkaleybar, Matin Hashemi, [gIM: GPU Accelerated RIS-based Influence Maximization Algorithm](), IEEE Transactions on Parallel and Distributed Systems (TPDS), 2021.

#### Abstract
Given a social network modeled as a weighted graph G, the influence maximization problem seeks k vertices to become initially influenced, to maximize the expected number of influenced nodes under a particular diffusion model. The influence maximization problem has been proven to be NP-hard, and most proposed solutions to the problem are approximate greedy algorithms, which can guarantee a tunable approximation ratio for their results with respect to the optimal solution. The state-of-the-art algorithms are based on Reverse Influence Sampling (RIS) technique, which can offer both computational efficiency and non-trivial (1-1/e-&#949)-approximation ratio guarantee for any &#949 > 0. RIS-based algorithms, despite their lower computational cost compared to other methods, still require long running times to solve the problem in large-scale graphs with low values of &#949. In this paper, we present a novel and efficient parallel implementation of a RIS-based algorithm, namely IMM, on GPU. The proposed GPU-accelerated influence maximization algorithm, named gIM, can significantly reduce the running time on large-scale graphs with low values of &#949. Furthermore, we show that gIM algorithm can solve other variations of the IM problem, only by applying minor modifications. Experimental results show that the proposed solution reduces the runtime by a factor up to 220x. The source code of gIM is publicly available online.

#### Citation
Please cite gIM in your publications if it helps your research:
```
@article{gim,
	author = {Soheil Shahrouz and Saber Salehkaleybar and Matin Hashemi}, 
	title = {gIM: GPU Accelerated RIS-based Influence Maximization Algorithm},
	journal = {IEEE Transactions on Parallel and Distributed Systems (TPDS)},
	year = {2021},
}
```
