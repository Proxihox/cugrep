/**
 * CUDA-accelerated pattern matching implementation of grep
 * Searches for patterns in text files using parallel GPU processing
 */

// Standard library includes
#include <iostream>
#include <fstream>
#include <queue>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>

// Configuration constants
#define MAX_MATCHES 60000    // Maximum number of matches to store
#define CHUNK_SIZE 400       // Size of text chunk per GPU thread
#define debug true          // Enable debug output

namespace fs = std::filesystem;
using namespace std;

// Fast I/O optimization
#define fastio ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);

// Global state
static bool complement = false;   // -v option: invert match
static bool recursive = false;    // -r option: recursive search
static bool case_insensitive = false; // -i option: case insensitive search
static bool match_start = false; //  match start of line
static bool match_end = false;   //  match end of line
static string pattern;           // Pattern to search for
static long long patternSize;    // Cached pattern length
static long long MAX_THREADS;    // Maximum GPU threads to use
static long long h_old;          // Track processed matches count

/**
 * Represents a pattern match location in text
 */
struct Match {
    long long start;    // Start position of match
    long long end;      // End position of match
};

/**
 * GPU configuration structure stored in constant memory
 */
struct GpuConfig {
    char* d_pattern;              // Device pattern string
    long long patternSize;        // Pattern length
    bool complement;              // Invert match flag
    Match *mlist;                // Array to store matches
    unsigned long long *d_counter; // Number of matches found
    bool case_insensitive; // Case insensitive flag
};

// GPU constant memory configuration
__constant__ GpuConfig c_config;

auto timenow(chrono::time_point<chrono::high_resolution_clock> startt, string s = "Time") {
    auto tnow = chrono::high_resolution_clock::now();
    if(debug) {
        fprintf(stderr, "%s : %ld us\n", 
                s.c_str(),
                chrono::duration_cast<chrono::microseconds>(tnow - startt).count());
    }
    return tnow;
}

/**
 * Handles result based on complement flag
 * @param a - Input value to complement if needed
 * @return Original or complemented boolean result
 */
long long result(long long a) {
    return complement ? !bool(a) : a;
}

/**
 * Searches for pattern within line using string comparison
 * @param line - Text line to search in
 * @return Position of match (1-based) or 0 if not found
 */
long long searchline(const string& line) {
    for(long long startPos = 0; startPos < line.size(); startPos++) {
        long long matchLen = 0;
        // Try matching pattern at current position
        while(startPos + matchLen < line.size() && 
              line[startPos + matchLen] == pattern[matchLen]) {
            matchLen++;
            if(matchLen == patternSize) {
                return result(startPos + 1);
            }
        }
    }
    return result(0);
}

/**
 * Processes input stream line by line
 * @param is - Input stream to process
 */
void searchstream(istream& is) {
    string line;
    while(getline(is, line)) {
        if(searchline(line)) {
            cout << line << '\n';
        }
    }
}



/**
 * Processes memory-mapped file data
 * @param data - Pointer to file data
 * @param size - Size of file in bytes
 */

__device__ long long deviceRes(long long a){
    return c_config.complement ? !bool(a) : a;
}

__device__ bool deviceMatch(const char* data, long long startPos){
    long long matchLen = 0;
    if(c_config.case_insensitive){
        while(data[startPos + matchLen] == c_config.d_pattern[matchLen] 
            or data[startPos + matchLen] == c_config.d_pattern[matchLen] - 32) {
            // Check for case-insensitive match
            matchLen++;
            if(matchLen == c_config.patternSize) {
                return deviceRes(startPos + 1);
            }
        }
    }
    else{
        while(data[startPos + matchLen] == c_config.d_pattern[matchLen]) {
            matchLen++;
            if(matchLen == c_config.patternSize) {
                return deviceRes(startPos + 1);
            }
        }
    }
    return deviceRes(0);
}

__device__ bool deviceMatch_CaseInsensitive(const char* data, long long startPos){
    long long matchLen = 0;
    while(data[startPos + matchLen] == c_config.d_pattern[matchLen] 
        or data[startPos + matchLen] == c_config.d_pattern[matchLen] - 32) {
        // Check for case-insensitive match
        matchLen++;
        if(matchLen == c_config.patternSize) {
            return deviceRes(startPos + 1);
        }
    }
    return deviceRes(0);
}


// Multiple Cuda Kernels, optimized for different patterns of search

// Basic string matching
__global__ void cudaSearchFile(const char* data, long long block_size, 
                                long long size_max, long long offset) {
    // Calculate thread's position and chunk
    long long id = threadIdx.x + 1024*blockIdx.x;
    long long iter = id*block_size;
    
    // Find start of current line
    while(iter > 0 && iter < size_max && data[iter] != '\n') iter--;
    
    long long limit = min((block_size)*(id+1), size_max);
    
    // Process lines in thread's chunk
    while(iter < limit) {
        long long startIndex = iter;
        bool matchFound = false;
        
        // Search current line for pattern
        while(iter < limit && iter < size_max && data[iter] != '\n') {
            if(!matchFound && deviceMatch(data, iter)) {
                matchFound = true;
            }
            iter++;
        }
        
        // Store match if found
        if(matchFound && iter < size_max && data[iter] == '\n') {
            long long index = atomicAdd(c_config.d_counter, 1);
            if(index < MAX_MATCHES) {
                c_config.mlist[index] = {offset+startIndex, offset+iter};
            }
        }
        iter++;
    }
}


__global__ void cudaSearchFile_Start(const char* data, long long block_size, 
                                long long size_max, long long offset) {
    // Calculate thread's position and chunk
    long long id = threadIdx.x + 1024*blockIdx.x;
    long long iter = id*block_size;
    
    // Find start of current line
    while(iter > 0 && iter < size_max && data[iter] != '\n') iter--;
    
    long long limit = min((block_size)*(id+1), size_max);
    
    // Process lines in thread's chunk
    while(iter < limit) {
        long long startIndex = iter;
        bool matchFound = false;
        matchFound = deviceMatch(data, startIndex);
        // Search current line for pattern
        while(iter < limit && iter < size_max && data[iter] != '\n') {
            iter++;
        }
        
        // Store match if found
        if(matchFound && iter < size_max && data[iter] == '\n') {
            long long index = atomicAdd(c_config.d_counter, 1);
            if(index < MAX_MATCHES) {
                c_config.mlist[index] = {offset+startIndex, offset+iter};
            }
        }
        iter++;
    }
}

__global__ void cudaSearchFile_End(const char* data, long long block_size, 
                                long long size_max, long long offset) {
    // Calculate thread's position and chunk
    long long id = threadIdx.x + 1024*blockIdx.x;
    long long iter = id*block_size;
    
    // Find start of current line
    while(iter > 0 && iter < size_max && data[iter] != '\n') iter--;
    
    long long limit = min((block_size)*(id+1), size_max);
    
    // Process lines in thread's chunk
    while(iter < limit) {
        long long startIndex = iter;
        bool matchFound = false;
        
        // Search current line for pattern
        while(iter < limit && iter < size_max && data[iter] != '\n') {
            iter++;
        }
        matchFound = deviceMatch(data, iter-c_config.patternSize);
        // Store match if found
        if(matchFound && iter < size_max && data[iter] == '\n') {
            long long index = atomicAdd(c_config.d_counter, 1);
            if(index < MAX_MATCHES) {
                c_config.mlist[index] = {offset+startIndex, offset+iter};
            }
        }
        iter++;
    }
}



void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

/**
 * Opens and memory-maps a file for processing
 * @param filePath - Path to file to process
 */
void fileMap(const string& filePath) {
    // Open file
    if(debug)fprintf(stderr, "File : %s\n", filePath.c_str());
    auto start = chrono::high_resolution_clock::now();
    const char* fname = filePath.c_str();
    long long fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return;
    }

    // Get file size
    struct stat sb;
    if(fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return;
    }

    // Memory map the file into program memory
    char* data = reinterpret_cast<char*>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0)
    );
    if(debug) start = timenow(start,"mmap");
    char* d_data;
    if(data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }
    
    // compute total number of threads
    long long total_threads = max(1ll,(long long)(sb.st_size/CHUNK_SIZE + 1));
    long long memiter = 0;
    if(debug) fprintf(stderr,"Total Memory : %lld\n",(long long)sb.st_size);
    if(debug) fprintf(stderr,"Total threads : %lld\n",total_threads);
    cudaError_t err = cudaMalloc(&d_data, min((long long)MAX_THREADS*CHUNK_SIZE, (long long)sb.st_size));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        munmap(data, sb.st_size);
        close(fd);
        return;
    }

    // Handle all threads, chunk by chunk
    while(total_threads > 0){

        long long n_threads = min(MAX_THREADS,total_threads);
        long long mem_end = min((long long) MAX_THREADS*CHUNK_SIZE + memiter,(long long)sb.st_size);
        long long memsize = min(mem_end - memiter, n_threads*CHUNK_SIZE);
        if(debug)start = timenow(start,"Setup");
        err = cudaMemcpy(d_data, data+memiter, memsize, cudaMemcpyHostToDevice);
        if(debug)start = timenow(start,"Memcpy");
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_data);
            munmap(data, sb.st_size);
            close(fd);
            return;
        }

        long long block_size = (sb.st_size+n_threads-1)/n_threads;
        long long n_blocks = (n_threads+1023)/1024;
        long long threads_per_block = min(1024ll,n_threads);
        if(match_start) {
            cudaSearchFile_Start<<<n_blocks,threads_per_block>>>(d_data,block_size,memsize,memiter);
        } else if(match_end) {
            cudaSearchFile_End<<<n_blocks,threads_per_block>>>(d_data,block_size,memsize,memiter);
        } else{
        cudaSearchFile<<<n_blocks,threads_per_block>>>(d_data,block_size,memsize,memiter);
        }
        checkCudaError("kernel launch");
        memiter += memsize;
        total_threads -= n_threads;
        // if(debug) fprintf(stderr, "threads : %lld\n", n_threads);
        // if(debug) fprintf(stderr,"Remaining threads : %lld\n",total_threads);
        // if(debug) fprintf(stderr,"Memsize : %lld\n",memsize);
        cudaDeviceSynchronize();
        start = timenow(start,"search");
        checkCudaError("kernel sync");
        if(debug)fprintf(stderr,"\n");
    }

    unsigned long long h_counter;
    Match *h_list = (Match*)malloc(sizeof(Match)*MAX_MATCHES);
    GpuConfig hostConfig;

    cudaMemcpyFromSymbol(&hostConfig, c_config, sizeof(GpuConfig));
    cudaMemcpy(&h_counter, hostConfig.d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    // Get matches from device
    cudaMemcpy(h_list, hostConfig.mlist, sizeof(Match)*h_counter, cudaMemcpyDeviceToHost);
    
    if(debug) fprintf(stderr, "Matches : %lld\n", h_counter);
    for(long long i = h_old; i < h_counter; i++){
        if(recursive){
            printf("%s:",fname);
        }
        for(long long j = max(1ll,h_list[i].start); j < h_list[i].end; j++){
            printf("%c",data[j]);
        }
        printf("\n");
        //printf(" %lld %lld %lld\n",h_list[i].start,h_list[i].end,i);
    }
    h_old = h_counter;
    free(h_list);
    start = timenow(start,"printing");
    munmap(data, sb.st_size);
    close(fd);
}

/**
 * Recursively processes files in directory
 * @param path - Starting directory path
 */
void iterFiles(const string& path) {
    try {
        for(const auto& entry : fs::recursive_directory_iterator(path)) {
            if(fs::is_regular_file(entry)) {
                fileMap(entry.path());
            }
        }
    } catch(const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << '\n';
    }
}

__host__ void initializeGPUMemory() {
    auto start = chrono::high_resolution_clock::now();
    // First CUDA call - will initialize GPU
    char* d_pattern;
    Match *mlist;
    unsigned long long *d_counter;
    cudaMalloc(&d_counter,sizeof(unsigned long long));
    // Add counter initialization
    unsigned long long init_val = 0;
    cudaMemcpy(d_counter, &init_val, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    
    cudaMalloc(&mlist,sizeof(Match)*MAX_MATCHES);
    cudaMalloc(&d_pattern, pattern.size() + 1);
    cudaMemcpy(d_pattern, pattern.c_str(), pattern.size() + 1, cudaMemcpyHostToDevice);
    GpuConfig hostConfig = {
        d_pattern,
        patternSize,
        complement,
        mlist,
        d_counter,
        case_insensitive
    };
    cudaMemcpyToSymbol(c_config, &hostConfig, sizeof(GpuConfig));

}

string pattern_init(string pattern) {
    if(pattern[0] == '^') {
        match_start = true;
        pattern = pattern.substr(1);
    }
    else if(pattern[pattern.size()-1] == '$') {
        match_end = true;
        pattern = pattern.substr(0, pattern.size()-1);
    }
    // Initialize pattern for case insensitive search
    if(case_insensitive) {
        for(auto& c : pattern) {
            c = tolower(c);
        }
    }
    return pattern;
}

__host__ void cleanupGPUMemory() {
    // Get config from constant memory
    GpuConfig hostConfig;
    cudaMemcpyFromSymbol(&hostConfig, c_config, sizeof(GpuConfig));
    
    // Free pattern memory
    if(hostConfig.d_pattern) {
        cudaFree(hostConfig.d_pattern);
    }
}

int main_handler(int argc, char *argv[]){
    auto startt = chrono::high_resolution_clock::now();
    fastio;
    startt = timenow(startt,"GPU Init");

    // Parse command line arguments
    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " [-rv] pattern [file...]\n";
        return 1;
    }

    // Parse options
    int argIndex = 1;
    if(argv[argIndex][0] == '-') {
        string args = argv[argIndex];
        for(size_t i = 1; i < args.length(); i++) {
            switch(args[i]) {
                case 'r': recursive = true; break;
                case 'v': complement = true; break;
                case 'i': case_insensitive = true; break;
                default:
                    cerr << "Unknown option: -" << args[i] << '\n';
                    return 1;
            }
        }
        argIndex++;
    }

    // Initialize pattern
    pattern = pattern_init(argv[argIndex++]);
    patternSize = pattern.size();
    initializeGPUMemory();
    h_old = 0;

    // Process input (stdin or files)
    if(argc == argIndex) {
        searchstream(cin);
    } else {
        for(int i = argIndex; i < argc; i++) {
            recursive ? iterFiles(argv[i]) : fileMap(argv[i]);
        }
    }
    cleanupGPUMemory();
    auto finalt = chrono::high_resolution_clock::now();
    fprintf(stderr, "TOTAL_TIME:%ld us\n", chrono::duration_cast<chrono::microseconds>(finalt - startt).count());
    return 0;
}

int main(int argc, char* argv[]){
    cudaFree(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Get properties of GPU 0
    MAX_THREADS = INT_MAX;
    if(argc == 1){ // Handle input stream
      string input;
      getline(cin, input);
      stringstream ss(input);
      string token;
      vector<string> tokens;
      while (ss >> token) {
        tokens.push_back(token);
      }
      char** new_argv = new char*[tokens.size() + 1];
      new_argv[0] = argv[0];
      for (size_t i = 0; i < tokens.size(); i++) {
        new_argv[i + 1] = const_cast<char*>(tokens[i].c_str());
      }
      argc = tokens.size() + 1;
      argv = new_argv;
      main_handler(argc,argv);
    }
    else main_handler(argc,argv); // Handle files
  }