#!/bin/bash
(cd ~/jetstream-pytorch && git pull)
 



# Define a function to cleanup on EXIT signal
cleanup() {
    sudo docker stop node
    sudo docker rm node
}
trap cleanup EXIT



sudo docker run \
    --entrypoint /bin/bash \
    --network host \
    -it \
    --name node \
    --shm-size 10.24g \
    --privileged \
    us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pytorch-server:v0.2.4 
    #  "${DOCKER_IMAGE}" -c "python examples/test_xla.py"
    # 

# git clone https://github.com/pytorch/xla.git

# use this to get into the container
# cmd bash /home/ohadr/vllm/examples/run_cluster.sh tpu-vm-base2 35.186.69.167 <hftoken> /dev/shm/huggingface
# docker exec -it node /bin/bash
# export  PT_XLA_DEBUG_LEVEL=2
# vllm serve meta-llama/Llama-3.1-8B-Instruct  --max-model-len 1024 --max-num-seqs 8  --distributed-executor-backend ray --tensor-parallel-size 4