#!/bin/bash
set -e

if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "ascend path: ${ASCEND_HOME_PATH}"
source $(dirname ${ASCEND_HOME_PATH})/set_env.sh

CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$CURRENT_DIR/output
echo "outpath: ${OUTPUT_DIR}"

COMPILE_OPTIONS=""

function build_deepep()
{
    CMAKE_DIR="csrc"
    BUILD_DIR="build"

    # 进入构建目录
    cd "$CMAKE_DIR" || exit

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" -DASCEND_HOME_PATH=$ASCEND_HOME_PATH -B "$BUILD_DIR" -S .
    # 编译并安装
    cmake --build "$BUILD_DIR" -j8 && cmake --build "$BUILD_DIR" --target install
    cd -
}

function make_package()
{
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel
    fi

    cp -v ${OUTPUT_DIR}/lib/* "$CURRENT_DIR"/deep_ep/
    rm -rf "$CURRENT_DIR"/dist
    python3 setup.py build
}

# 编译安装算子
function build_kernels()
{
    KERNEL_DIR="csrc/kernels"
    # 算子包安装路径
    CUSTOM_OPP_DIR="${CURRENT_DIR}/deep_ep"

    # 进入构建目录
    cd "$KERNEL_DIR" || exit

    # 编译自定义算子
    chmod +x build.sh
    # 初次编译会因为权限报错 
    chmod +x cmake/util/gen_ops_filter.sh
    ./build.sh

    # 安装自定义算子
    custom_opp_file=$(find ./build_out -maxdepth 1 -type f -name "custom_opp*.run")
    if [ -z "$custom_opp_file" ]; then
        echo "can not find run package"
        exit 1
    else
        # 执行找到的文件
        echo "find run package: $custom_opp_file"
        chmod +x "$custom_opp_file"  # 确保可执行权限
    fi
    # echo "./build_out/custom_opp_*.run --install-path=$CUSTOM_OPP_DIR"
    ./build_out/custom_opp_*.run --install-path=$CUSTOM_OPP_DIR

    cd -
}


function main()
{
    # 构建自定义算子并安装
    # build_kernels
    # 构建适配层动态库 deep_ep_cpp.cpython-39-aarch64-linux-gnu.so
    build_deepep
    # 打包python package
    make_package
}

main