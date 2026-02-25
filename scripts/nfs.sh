#!/bin/bash

nvme_device="/dev/nvme0n1"
nvme_mnt_path="/mnt/nvme"
# 不再需要指定特殊端口，NFS默认使用2049

nfs_server_setup(){
    echo "Setting up NFS server..."
    
    # 安装NFS服务器
    sudo apt-get update
    sudo apt-get install -y nfs-kernel-server
    
    # 格式化并挂载NVMe设备
    echo "Formatting ${nvme_device}..."
    sudo mkfs.ext4 -F ${nvme_device}
    
    echo "Mounting ${nvme_device} to ${nvme_mnt_path}..."
    sudo mkdir -p ${nvme_mnt_path}
    sudo mount -o defaults ${nvme_device} ${nvme_mnt_path}
    
    # 配置自动挂载
    echo "${nvme_device} ${nvme_mnt_path} ext4 defaults 0 0" | sudo tee -a /etc/fstab
    
    # 配置NFS导出
    echo "Configuring NFS export..."
    echo "${nvme_mnt_path} *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee /etc/exports
    
    # 重启NFS服务
    sudo systemctl restart nfs-server || sudo service nfs-kernel-server restart
    
    # 显示导出信息
    echo ""
    echo "NFS Server setup completed!"
    echo "Exported directory: ${nvme_mnt_path}"
    echo "To check exports: sudo exportfs -v"
    echo ""
    echo "On client, mount with:"
    echo "sudo mount -t nfs $(hostname -I | awk '{print $1}'):${nvme_mnt_path} /mnt/nvme"
}

nfs_client_setup(){
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 client <server_ip>"
        echo "Example: $0 client 10.118.0.166"
        exit 1
    fi
    
    nvme_ip="$2"
    
    echo "Setting up NFS client..."
    
    # 安装NFS客户端
    sudo apt-get update
    sudo apt-get install -y nfs-common
    
    # 创建挂载点
    echo "Creating mount point ${nvme_mnt_path}..."
    sudo mkdir -p ${nvme_mnt_path}
    
    # 挂载NFS共享（使用NFSv4，TCP协议）
    echo "Mounting NFS share from ${nvme_ip}:${nvme_mnt_path}..."
    sudo mount -t nfs -o vers=4,proto=tcp ${nvme_ip}:${nvme_mnt_path} ${nvme_mnt_path}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "NFS Client setup completed!"
        echo "Mounted: ${nvme_ip}:${nvme_mnt_path} to ${nvme_mnt_path}"
        echo "Check with: df -h ${nvme_mnt_path}"
        
        # 配置自动挂载（可选）
        echo ""
        echo "To auto-mount on boot, add to /etc/fstab:"
        echo "${nvme_ip}:${nvme_mnt_path} ${nvme_mnt_path} nfs vers=4,proto=tcp 0 0"
    else
        echo "ERROR: Failed to mount NFS share"
        echo "Check:"
        echo "1. Server IP is correct"
        echo "2. Server NFS is running: sudo showmount -e ${nvme_ip}"
        echo "3. Firewall allows NFS traffic"
    fi
}

check_nfs_server(){
    echo "Checking NFS server status..."
    sudo systemctl status nfs-server || sudo service nfs-kernel-server status
    echo ""
    echo "Current exports:"
    sudo exportfs -v
}

# 主执行逻辑
case "$1" in
    "server")
        nfs_server_setup
        ;;
    "client")
        if [ -z "$2" ]; then
            echo "Error: Server IP required for client setup"
            echo "Usage: $0 client <server_ip>"
            echo "Example: $0 client 10.118.0.166"
            exit 1
        fi
        nfs_client_setup "$@"
        ;;
    "check")
        check_nfs_server
        ;;
    *)
        echo "Usage: $0 <server|client <server_ip>|check>"
        echo ""
        echo "Examples:"
        echo "  $0 server                    # Setup NFS server"
        echo "  $0 client 10.118.0.166      # Setup NFS client (connect to server)"
        echo "  $0 check                    # Check NFS server status"
        exit 1
        ;;
esac