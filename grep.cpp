#include <iostream>
#include <fstream>
#include <queue>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>

// Fast I/O optimization
#define fastio ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);

namespace fs = std::filesystem;
using namespace std;

// Global configuration
static bool complement = false;   // -v option: invert match
static bool recursive = false;    // -r option: recursive search
static string pattern;           // Search pattern
static int patternSize;         // Pattern length cached for performance

/**
 * Handles result based on complement flag
 * @param a - Input value to complement if needed
 * @return Original or complemented boolean result
 */
int result(int a) {
    return complement ? !bool(a) : a;
}

/**
 * Searches for pattern within line using string comparison
 * @param line - Text line to search in
 * @return Position of match (1-based) or 0 if not found
 */
int searchline(const string& line) {
    for(int startPos = 0; startPos < line.size(); startPos++) {
        int matchLen = 0;
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
 * Memory-mapped file pattern matching
 * @param data - Pointer to memory-mapped file data
 * @param startPos - Starting position in data
 * @return True if pattern found at position
 */
bool match(const char* data, int startPos) {
    int matchLen = 0;
    while(data[startPos + matchLen] == pattern[matchLen]) {
        matchLen++;
        if(matchLen == patternSize) {
            return result(startPos + 1);
        }
    }
    return result(0);
}

/**
 * Processes memory-mapped file data
 * @param data - Pointer to file data
 * @param size - Size of file in bytes
 */
void searchFile(const char* data, int size,string fpath = NULL) {
    int iter = 0;
    while(iter < size) {
        int startIndex = iter;
        bool matchFound = false;
        
        // Process current line
        while(data[iter] != '\n' && iter < size) {
            if(!matchFound && match(data, iter)) {
                matchFound = true;
            }
            iter++;
        }
        
        // Output matching line
        if(matchFound) {
            if(recursive) cout << fpath << ":";
            cout.write(data + startIndex, iter - startIndex);
            cout << '\n';
        }
        iter++;
    }
}

/**
 * Timing helper function
 * @param startt - Start time point
 * @param s - Description string
 * @return Current time point
 */
auto timenow(std::chrono::time_point<std::chrono::high_resolution_clock> startt, string s = "Time") {
    auto tnow = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s : %ld us\n", 
            s.c_str(),
            std::chrono::duration_cast<std::chrono::microseconds>(tnow - startt).count());
    return tnow;
}

/**
 * Opens and memory-maps a file for processing
 * @param filePath - Path to file to process
 */
void fileMap(const string& filePath) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Open file
    const char* fname = filePath.c_str();
    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return;
    }
    start = timenow(start, "File open");

    // Get file size
    struct stat sb;
    if(fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return;
    }
    fprintf(stderr, "File size: %ld bytes\n", sb.st_size);

    // Memory map the file
    start = timenow(start, "Stats");
    char* data = reinterpret_cast<char*>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0)
    );
    if(data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }
    start = timenow(start, "Memory map");

    // Process file
    searchFile(data, sb.st_size,fname);
    start = timenow(start, "Search");

    // Cleanup
    munmap(data, sb.st_size);
    close(fd);
    start = timenow(start, "Cleanup");
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

int main(int argc, char *argv[]) {
    auto total_start = std::chrono::high_resolution_clock::now();
    fastio;

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
                default:
                    cerr << "Unknown option: -" << args[i] << '\n';
                    return 1;
            }
        }
        argIndex++;
    }

    // Initialize pattern
    pattern = argv[argIndex++];
    patternSize = pattern.size();

    // Process input (stdin or files)
    if(argc == argIndex) {
        searchstream(cin);
    } else {
        for(int i = argIndex; i < argc; i++) {
            recursive ? iterFiles(argv[i]) : fileMap(argv[i]);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "TOTAL_TIME:%ld\n", 
            std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count());
    return 0;
}