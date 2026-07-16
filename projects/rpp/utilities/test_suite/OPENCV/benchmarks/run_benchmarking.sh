#!/bin/bash

# MIT License
#
# Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Complete setup script for OpenCV benchmark
# This script handles everything from installation to running the benchmark

set -e

# Default values
NUM_THREADS=""
NUM_RUNS=""
CLEAN_BUILD=1  # Default to fresh build

# Parse command-line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threads <N>     Number of threads to use (default: auto-detect)"
    echo "  -n, --num-runs <N>    Number of benchmark runs (default: 100)"
    echo "  --no-clean            Skip clean build (use existing build)"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Fresh build, auto-detect threads, 100 runs (default)"
    echo "  $0 -t 64               # Fresh build with 64 threads"
    echo "  $0 -t 32 -n 50         # Fresh build with 32 threads, 50 runs"
    echo "  $0 --no-clean          # Use existing build without cleaning"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        -n|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --no-clean)
            CLEAN_BUILD=0
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "OpenCV Benchmark - Complete Setup"
echo "========================================"
echo ""
if [ -n "$NUM_THREADS" ] || [ -n "$NUM_RUNS" ]; then
    echo "Configuration:"
    [ -n "$NUM_THREADS" ] && echo "  Threads: $NUM_THREADS"
    [ -n "$NUM_RUNS" ] && echo "  Number of runs: $NUM_RUNS"
    echo ""
fi

# Helper function: Run command with sudo if not root
run_cmd() {
    if [ "$EUID" -ne 0 ]; then
        sudo "$@"
    else
        "$@"
    fi
}

# Helper function to uninstall OpenCV 4.6.0
uninstall_opencv_4() {
    echo ""
    echo "Uninstalling OpenCV 4.6.0 system packages..."
    echo "--------------------------------------"

    run_cmd apt-get remove --purge -y \
        libopencv-calib3d-dev libopencv-calib3d406t64 \
        libopencv-contrib-dev libopencv-contrib406t64 \
        libopencv-core-dev libopencv-core406t64 \
        libopencv-dev \
        libopencv-dnn-dev libopencv-dnn406t64 \
        libopencv-features2d-dev libopencv-features2d406t64 \
        libopencv-flann-dev libopencv-flann406t64 \
        libopencv-highgui-dev libopencv-highgui406t64 \
        libopencv-imgcodecs-dev libopencv-imgcodecs406t64 \
        libopencv-imgproc-dev libopencv-imgproc406t64 \
        libopencv-java \
        libopencv-ml-dev libopencv-ml406t64 \
        libopencv-objdetect-dev libopencv-objdetect406t64 \
        libopencv-photo-dev libopencv-photo406t64 \
        libopencv-shape-dev libopencv-shape406t64 \
        libopencv-stitching-dev libopencv-stitching406t64 \
        libopencv-superres-dev libopencv-superres406t64 \
        libopencv-video-dev libopencv-video406t64 \
        libopencv-videoio-dev libopencv-videoio406t64 \
        libopencv-videostab-dev libopencv-videostab406t64 \
        libopencv-viz-dev libopencv-viz406t64 \
        libopencv406-jni 2>/dev/null || true

    run_cmd apt-get autoremove -y
    run_cmd rm -rf /usr/lib/x86_64-linux-gnu/cmake/opencv4
    run_cmd rm -f /usr/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc
    run_cmd rm -rf /usr/include/opencv4

    echo "✓ OpenCV 4.6.0 completely removed"
    echo ""
}

# Helper function to install OpenCV 5.0.0
install_opencv_5() {
    local VERSION="$1"

    echo "Installing OpenCV $VERSION with required components..."
    run_cmd apt-get install -y wget unzip

    # Clean /usr/local
    echo "Cleaning /usr/local from previous OpenCV installations..."
    run_cmd rm -rf /usr/local/lib/cmake/opencv5
    run_cmd rm -f /usr/local/lib/libopencv_*
    run_cmd rm -rf /usr/local/include/opencv5
    run_cmd rm -f /usr/local/lib/pkgconfig/opencv4.pc

    # Build OpenCV from source
    cd /tmp
    wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${VERSION}.zip
    unzip -q opencv.zip
    cd opencv-${VERSION}
    mkdir -p build && cd build

    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_core=ON \
          -D BUILD_opencv_imgproc=ON \
          -D BUILD_opencv_imgcodecs=ON \
          -D BUILD_opencv_calib3d=ON \
          -D BUILD_opencv_features2d=ON \
          -D BUILD_opencv_flann=ON \
          .. > /dev/null

    make -j$(nproc)
    run_cmd make install
    run_cmd ldconfig
    cd "$SCRIPT_DIR"
    rm -rf /tmp/opencv.zip /tmp/opencv-${VERSION}

    echo "✓ OpenCV $VERSION installed successfully"
}

# Check OpenCV version and handle upgrades
OPENCV_VERSION_REQUIRED="5.0.0"
OPENCV_INSTALLED=0
NEED_UNINSTALL_4=0

# Check if OpenCV 4.6.0 is installed from apt
if dpkg -l 2>/dev/null | grep -q "^ii.*libopencv-dev.*4\.6\.0"; then
    echo "⚠ Found OpenCV 4.6.0 from system packages - will uninstall before installing 5.0.0"
    NEED_UNINSTALL_4=1
    OPENCV_INSTALLED=0
# Check if OpenCV 5.0.0 is already installed in /usr/local
elif [ -f "/usr/local/lib/libopencv_core.so.${OPENCV_VERSION_REQUIRED}" ] && \
     [ -f "/usr/local/lib/libopencv_imgproc.so.${OPENCV_VERSION_REQUIRED}" ] && \
     [ -f "/usr/local/lib/libopencv_imgcodecs.so.${OPENCV_VERSION_REQUIRED}" ] && \
     [ -f "/usr/local/lib/libopencv_calib.so.${OPENCV_VERSION_REQUIRED}" ]; then
    OPENCV_INSTALLED=1
    echo "✓ OpenCV $OPENCV_VERSION_REQUIRED already installed with required components"
else
    echo "OpenCV $OPENCV_VERSION_REQUIRED not found or missing required components"
    OPENCV_INSTALLED=0
fi

# Check if other dependencies are installed
DEPS_MISSING=0
if ! dpkg -l | grep -q libxlsxwriter-dev; then
    DEPS_MISSING=1
fi

# Install dependencies if needed
if [ $OPENCV_INSTALLED -eq 0 ] || [ $DEPS_MISSING -eq 1 ]; then
    echo "Step 1: Installing system dependencies..."
    echo "--------------------------------------"
    echo "This requires sudo privileges."
    echo ""

    [ "$EUID" -ne 0 ] && echo "Please enter your password to install dependencies:"

    # Uninstall OpenCV 4.6.0 if present
    [ $NEED_UNINSTALL_4 -eq 1 ] && uninstall_opencv_4

    # Install basic dependencies
    run_cmd apt-get update
    run_cmd apt-get install -y libgomp1 cmake build-essential libxlsxwriter-dev python3-pip

    # Install OpenCV 5.0.0 if not present
    [ $OPENCV_INSTALLED -eq 0 ] && install_opencv_5 "$OPENCV_VERSION_REQUIRED"

    echo ""
else
    echo "✓ System dependencies already installed"
    echo "  OpenCV Version: 5.0.0"
    echo "  libxlsxwriter: Installed"
    echo ""
fi

# Install Python dependencies
echo "Step 2: Installing Python dependencies..."
echo "--------------------------------------"
if ! python3 -c "import PIL" 2>/dev/null; then
    pip3 install --user Pillow
else
    echo "✓ Pillow already installed"
fi
echo ""

# Check if dataset exists
if [ ! -d "input_images_dataset" ] || [ -z "$(ls -A input_images_dataset 2>/dev/null)" ]; then
    echo "Step 3: Generating test dataset..."
    echo "--------------------------------------"
    python3 generate_test_dataset.py
    echo ""
else
    echo "✓ Dataset already exists ($(ls -1 input_images_dataset | wc -l) images)"
    echo ""
fi

# Build benchmark
echo "Step 4: Building benchmark..."
echo "--------------------------------------"
if [ $CLEAN_BUILD -eq 1 ]; then
    echo "Performing fresh build (cleaning old build artifacts)..."
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
    echo "✓ Fresh build complete"
else
    echo "Using existing build (incremental build)..."
    mkdir -p build
    cd build
    [ ! -f Makefile ] && cmake ..
    make -j$(nproc)
    cd ..
    echo "✓ Incremental build complete"
fi
echo ""

# Build command with optional arguments
BUILD_DIR="build"
cmd="./${BUILD_DIR}/opencv_vs_rpp_host_hip_benchmarking"
cmd_args=""

[ -n "$NUM_THREADS" ] && cmd_args="$cmd_args --threads $NUM_THREADS"
[ -n "$NUM_RUNS" ] && cmd_args="$cmd_args --num-runs $NUM_RUNS"

runs_text="${NUM_RUNS:-100}"
threads_text="${NUM_THREADS:-auto-detect}"

echo "========================================"
echo "Starting Benchmark"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Threads: $threads_text"
echo "  Runs: $runs_text"
echo ""
echo "This will take several minutes..."
echo ""

$cmd $cmd_args

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
