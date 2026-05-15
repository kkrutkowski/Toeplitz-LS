#!/bin/bash
# Generate the build system in ./tmp
cmake -B tmp -S .

# Compile the code
cmake --build tmp
