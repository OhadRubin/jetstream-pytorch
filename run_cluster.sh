#!/bin/bash

(cd ~/jetstream-pytorch && git pull)
# Get the current IP address
CURRENT_IP=$(curl https://checkip.amazonaws.com)
echo "Current IP address: ${CURRENT_IP}"

# Check for minimum number of required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_address hf_token path_to_hf_home [additional_args...]"
    exit 1
fi

 

# Assign the first three arguments and shift them away
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
HF_TOKEN="$3"  # Should be --head or --worker
PATH_TO_HF_HOME="$4"
shift 4

# Additional arguments are passed directly to the Docker command
ADDITIONAL_ARGS=("$@")


# Define a function to cleanup on EXIT signal
cleanup() {
    sudo docker stop node
    sudo docker rm node
}
trap cleanup EXIT

# Command setup for head or worker node
RAY_START_CMD="ray start --block --num-cpus=220 --resources='{\"TPU\": 4}'"
if [ "${CURRENT_IP}" == "${HEAD_NODE_ADDRESS}" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# cmd sudo docker build -t tpu-vm-base2 -f Dockerfile.tpu .
# Run the docker command with the user specified parameters and additional arguments

# docker run -v $(pwd):/workspace/jetstream-pytorch -it your-image-name


# -it \
sudo docker run \
    --entrypoint /bin/bash \
    --network host \
    --name node \
    --shm-size 10.24g \
    --privileged \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e GLOO_SOCKET_IFNAME=ens8 \
    -v /mnt/ramdisk:/mnt/ramdisk \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "cd /jetstream-pytorch  &&  ${RAY_START_CMD}"
# git clone https://github.com/pytorch/xla.git

# use this to get into the container
# cmd bash /home/ohadr/jetstream-pytorch/examples/run_cluster.sh tpu-vm-base2 35.186.69.167 <hftoken> /dev/shm/huggingface
# sudo docker exec -it node /bin/bash
# export  PT_XLA_DEBUG_LEVEL=2
# jetstream-pytorch serve meta-llama/Llama-3.1-8B-Instruct  --max-model-len 1024 --max-num-seqs 8  --distributed-executor-backend ray --tensor-parallel-size 4

# ray job submit --runtime-env-json='{"working_dir": "."}' -- python run_ray_serve_interleave.py  --tpu_chips=4 --num_hosts=1 --size=7b --model_name=llama-2 --batch_size=32 --max_cache_length=2048 --tokenizer_path=/mnt/ramdisk/Llama-3.1-8B-Instruct/tokenizer.json --checkpoint_path=/mnt/ramdisk/Llama-3.1-8B-Instruct/ --quantize_weights=True --quantize_type="int8_per_channel" --quantize_kv_cache=True --sharding_config="default_shardings/llama.yaml"

# cd /jetstream-pytorch
# export DISABLE_XLA2_PJRT_TEST="true"
# python3 run_server_with_ray.py --tpu_chips=8 --num_hosts=2 --worker_chips=4 --model_name=llama-3 --size=7b --batch_size=30 --max_cache_length=2048 --sharding_config="default_shardings/llama.yaml" --tokenizer_path=/mnt/ramdisk/Llama-3.1-8B-Instruct/tokenizer.json --checkpoint_path=/mnt/ramdisk/Llama-3.1-8B-Instruct/ 

# --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path 


