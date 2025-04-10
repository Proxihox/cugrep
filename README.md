# Cuda Accelerated Grep

## Requirements
You will need access to an Nvidia GPU and have `nvcc` (cuda compiler) installed on a Linux based OS.
## Setup
Clone the repository and run `make` to compile. It will generate 2 binaries, `mygrep` and `cugrep`. `cugrep` is the GPU accelerated version.
To run you will need to use `<path_to_ccat>/cugrep`. If you wish to call it using just `cugrep`, move the compiled binary into `/usr/local/bin` using 
```
sudo mv cugrep /usr/local/bin/
```

Supported Arguments:

| Arg | Function |
|:---:|:-------- |
| -i | Case insensitive search |
| -v | Invert match |
| -r | Recursive |

Currently support for only one argument at a time is available.
