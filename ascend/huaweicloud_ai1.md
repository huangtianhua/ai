# 在华为云上使用Ai1实例部署Ascend开发环境

华为云目前提供了基于Ascend 310的虚拟机，实例类型为Ai1，支持的OS为ubuntu 18.04和centos 7.6。本文以ubuntu 18.04的4U16G并搭载两块ascend 310的VM为例，说明如何初始化开发环境。

1. 卸载旧版固件

    华为云的Ai1实例默认已经安装了Ascend的驱动和lib库，但版本可能已过时，如果想使用新版的开发环境，则需要先删除旧版固件。

    ```shell
    # 进入/usr/local/Ascend，
    cd /usr/local/Ascend
    #按照opp、ascend-tookit、driver的顺序，依次进入相关目录的script目录中，执行uninstall.sh文件
    ```

    卸载完成后，需要重启VM

2. 安装驱动和开发库

    从昇腾官方下载最新的驱动和开发包

    - [CANN-NNRT](https://www.hiascend.com/software/cann/community)(NNRT是推理运行时包，开发环境可以不用安装，已被toolkit集成）
    - [CANN-TOOLKIT](https://www.hiascend.com/software/cann/community)
    - [Ascend 310 driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community)

    下载对应的`.run`包，拷贝到VM中，赋予执行权限，按照driver、nnrt、toolkit的顺序依此安装。

    ```shell
    # 安装driver
    ./A300-3010-npu-driver_21.0.4_linux-x86_64.run --full
    # 检查安装结果，显示两块OK的ascend 310卡
    npu-smi info
    # 安装nnrt
    ./Ascend-cann-nnrt_5.1.RC2.alpha002_linux-x86_64.run --install：
    # 安装依赖
    apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev
    # 升级pip
    pip3 install --upgrade pip
    # 安装pip依赖
    pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests
    # 安装toolkit
    ./Ascend-cann-toolkit_5.1.RC2.alpha002_linux-x86_64.run --install
    ```

3. 测试

    按照[官方指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha002/infacldevg/aclcppdevg/aclcppdevg_000001.html),下载example工程并编译执行即可。
