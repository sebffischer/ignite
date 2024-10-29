#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display messages
echo_message() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# 1. Run cmake for the csrc directory
echo_message "Running cmake for csrc..."

cd csrc/build

cmake ..

# 2. Build and package the project using cmake with 8 parallel jobs
echo_message "Building and packaging with cmake..."
cmake --build . --target package --parallel 8

# 3. Remove the existing build artifact using `trash` only if it exists
BUILD_ARTIFACT="ignite-0.0.0.9000+cpu-Darwin"
if [ -e "$BUILD_ARTIFACT" ]; then
    echo_message "Removing existing build artifact..."
    trash "$BUILD_ARTIFACT"
else
    echo_message "Build artifact '$BUILD_ARTIFACT' does not exist. Skipping trash command."
fi

# 4. Unzip the packaged build
echo_message "Unzipping the packaged build..."
unzip ignite-0.0.0.9000+cpu-Darwin.zip

# 5. Copy ignite.def to inst/def
echo_message "Copying ignite.def to inst/def..."
cp ../src/ignite.def ../../inst/def

# 6. Copy include files to inst/include/
echo_message "Copying include files to inst/include/..."
cp -r ../include/ignite ../../inst/include/

cd ../..

echo_message "Build process completed successfully!"

Rscript -e 'Rcpp::compileAttributes(); devtools::install(".", upgrade = "never")'
