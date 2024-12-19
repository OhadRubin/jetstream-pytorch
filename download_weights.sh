
HF_TOKEN="$1"
MODEL_NAME="$2"
# meta-llama/Llama-3.1-8B-Instruct
sudo mkdir -p /mnt/ramdisk
if ! mountpoint -q /mnt/ramdisk; then
  sudo mount -t tmpfs tmpfs /mnt/ramdisk
  sudo chmod 777 /mnt/ramdisk
fi


huggingface-cli login --token $HF_TOKEN
huggingface-cli download --repo-type model --repo-id $MODEL_NAME --local-dir /mnt/ramdisk/$MODEL_NAME

