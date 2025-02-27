g++ -O3 -o mygrep grep.cpp 
nvcc -O3 -o cugrep grep.cu
# Run CPU and GPU versions with recursive flag and compare outputs
./mygrep -r /testdata > cpu_output.txt
./cugrep -r /testdata > gpu_output.txt
diff cpu_output.txt gpu_output.txt

# Compare with system grep as reference
grep -r /testdata > grep_output.txt
diff cpu_output.txt grep_output.txt
diff gpu_output.txt grep_output.txt