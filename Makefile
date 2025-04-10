all : cugrep mygrep

cugrep : grep.cpp
	g++ -O3 -o mygrep grep.cpp

mygrep : grep.cu
	nvcc -O3 -o cugrep grep.cu
