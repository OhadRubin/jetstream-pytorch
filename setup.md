```sh
sudo docker run \
-v /home/$USER/vllm:/workspace/vllm \
--entrypoint /bin/bash \
--network host \
--name node \
--shm-size 10.24g \
--privileged \
-e GLOO_SOCKET_IFNAME=ens8 \
us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pytorch-server:v0.2.4 \
-c "cd /workspace/vllm && \
--model_id=meta-llama/Meta-Llama-3-8B \
--override_batch_size=30 \
--working_dir=/models/pytorch/ \
--enable_model_warmup=True"


sudo  docker pull us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pytorch-server:v0.2.4

sudo docker run us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pytorch-server:v0.2.4 --entrypoint /bin/bash
```

jetstream_pytorch_server_entrypoint..sh
```sh

export HUGGINGFACE_TOKEN_DIR="/huggingface"
cd /jetstream-pytorch
huggingface-cli login --token $(cat ${HUGGINGFACE_TOKEN_DIR}/HUGGINGFACE_TOKEN)
jpt serve --model_id=meta-llama/Meta-Llama-3-8B --override_batch_size=30 --working_dir=/models/pytorch/ --enable_model_warmup=True

```