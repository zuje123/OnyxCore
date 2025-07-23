# OnyxCore

OnyxCore（ˈɑnɪks kɔr）
Onyx → ​​Overlapping NPU​​（黑曜石的层叠纹理=>计算与通信的时空咬合）
Core → ​​Kernel Acceleration​​（核心算力引擎）


## Quick start

### Requirements
硬件型号支持：Atlas A3 系列产品
平台：aarch64/x86
配套软件
- 驱动固件 Ascend HDK 25.0.RC1.1、CANN 8.1.RC1及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&amp;InstallType=local&amp;OS=Ubuntu&amp;Software=cannToolKit)》安装CANN开发套件包以及配套固件和驱动）
- 安装CANN软件前需安装相关[依赖列表](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)

### Development
1、准备CANN的环境变量（根据安装路径修改）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、构建项目
执行工程构建脚本 build.sh前，根据CANN安装路径，修改`build.sh:line7`的`_ASCEND_INSTALL_PATH`。
```bash
# 构建项目
bash build.sh

# 根据你的设置软链接到 deep_ep_cpp.*.so 文件
ln -s build/lib.linux-aarch64-cpython-39/deep_ep/deep_ep_cpp.cpython-39-aarch64-linux-gnu.so

# 运行测试用例
python3 tests/test_buffer.py
```
说明：
```bash
# build.sh:line84
function main()
{
    # 构建自定义算子并安装
    build_kernels
    # 构建适配层动态库 deep_ep_cpp.cpython-39-aarch64-linux-gnu.so
    build_deepep
    # 打包python package
    make_package
}
```
构建过程主要分为3步，在开发过程中可以根据需要注释部分步骤。如果未修改可注释`build_kernels`，`csrc`的适配层未修改可注释`build_ascendep`。


### Installation
1、执行安装脚本，将`.whl`安装到你的python环境下
```bash
bash install.sh
```
2、执行CANN的环境变量（根据安装路径修改）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
3、在python工程中导入`deep_ep`


### FAQ
1、在带算子构建时，初次编译可能会因为某个`.sh`的权限问题报错，添加执行权限即可；
![build_error](figures/build_error.png)
添加权限：
```bash
chmod +x csrc/kernels/cmake/util/gen_ops_filter.sh
```
