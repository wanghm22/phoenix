#!/bin/bash

nvme_subsystem_name="nvme-subsystem-name"
namespaces=10
nvme_device="/dev/nvme0n1"
nvme_port=1
nvme_ip="10.118.0.166"
nvme_ip_port=4420

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <target|initiator> <setup|cleanup>"
    exit 1
fi

mlex_model_probe(){
    modprobe nvmet
    modprobe nvmet-tcp  # 改为TCP模块
    modprobe nvme-tcp   # 改为TCP模块
}

nvme_of_target_setup(){
    # 加载必要的内核模块
    mlex_model_probe
    
    # 创建子系统
    sudo mkdir -p /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name} || exit 1
    cd /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name} || exit 1
    echo 1 | sudo tee attr_allow_any_host > /dev/null
    
    # 创建命名空间
    sudo mkdir -p namespaces/${namespaces} || exit 1
    cd namespaces/${namespaces} || exit 1
    echo -n ${nvme_device} | sudo tee device_path > /dev/null
    echo 1 | sudo tee enable > /dev/null
    
    # 创建端口
    sudo mkdir -p /sys/kernel/config/nvmet/ports/${nvme_port} || exit 1
    cd /sys/kernel/config/nvmet/ports/${nvme_port} || exit 1
    echo "${nvme_ip}" | sudo tee addr_traddr > /dev/null
    echo tcp | sudo tee addr_trtype > /dev/null  # 改为tcp
    echo ${nvme_ip_port} | sudo tee addr_trsvcid > /dev/null
    echo ipv4 | sudo tee addr_adrfam > /dev/null
    
    # 链接子系统到端口 - 使用变量而不是硬编码的"1"
    sudo ln -sf /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name} \
         /sys/kernel/config/nvmet/ports/${nvme_port}/subsystems/${nvme_subsystem_name}
    
    echo "NVMe-oF Target setup completed successfully (TCP mode)"
}

nvme_of_initiator_setup(){
    sudo nvme discover -t tcp -q ${nvme_subsystem_name} -a ${nvme_ip} -s ${nvme_ip_port}  # 改为tcp
    sudo nvme connect -t tcp -q ${nvme_subsystem_name} -n ${nvme_subsystem_name} -a ${nvme_ip} -s ${nvme_ip_port}  # 改为tcp
}

nvme_of_initiator_cleanup(){
    sudo nvme disconnect -n ${nvme_subsystem_name}
}

nvme_of_target_cleanup(){
    # 移除链接
    sudo rm -f /sys/kernel/config/nvmet/ports/${nvme_port}/subsystems/${nvme_subsystem_name}
    
    # 禁用并移除命名空间
    if [ -d "/sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}/namespaces/${namespaces}" ]; then
        echo 0 | sudo tee /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}/namespaces/${namespaces}/enable > /dev/null
        sudo rmdir /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}/namespaces/${namespaces}
    fi
    
    # 移除子系统
    if [ -d "/sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}" ]; then
        sudo rmdir /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}
    fi
    
    # 移除端口
    if [ -d "/sys/kernel/config/nvmet/ports/${nvme_port}" ]; then
        sudo rmdir /sys/kernel/config/nvmet/ports/${nvme_port}
    fi
    
    echo "NVMe-oF Target cleanup completed"
}

check_nvme_cli(){
    if ! command -v nvme &> /dev/null
    then
        echo "nvme command could not be found"
        echo "Install with: apt-get install nvme-cli"
        exit 1
    fi
}

# 主执行逻辑
case "$1" in
    "target")
        if [ "$2" == "setup" ]; then
            nvme_of_target_setup
        elif [ "$2" == "cleanup" ]; then
            nvme_of_target_cleanup
        else
            echo "Usage: $0 target <setup|cleanup>"
        fi
        ;;
    "initiator")
        check_nvme_cli
        if [ "$2" == "setup" ]; then
            nvme_of_initiator_setup
        elif [ "$2" == "cleanup" ]; then
            nvme_of_initiator_cleanup
        else
            echo "Usage: $0 initiator <setup|cleanup>"
        fi
        ;;
    *)
        echo "Usage: $0 <target|initiator> <setup|cleanup>"
        exit 1
        ;;
esac