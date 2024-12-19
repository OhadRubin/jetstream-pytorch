
HF_TOKEN="$1"
MODEL_NAME="$2"
OUTPUT_DIR="$3"
shift 3
ADDITIONAL_ARGS=("$@")
# usage: ./download_weights.sh <HF_TOKEN> <MODEL_NAME>
# example: ./download_weights.sh <HF_TOKEN> meta-llama/Llama-3.1-8B-Instruct
sudo mkdir -p /mnt/ramdisk
if ! mountpoint -q /mnt/ramdisk; then
  sudo mount -t tmpfs tmpfs /mnt/ramdisk
  sudo chmod 777 /mnt/ramdisk
fi


huggingface-cli download --token $HF_TOKEN --exclude "*original*" --local-dir /mnt/ramdisk/$OUTPUT_DIR  $ADDITIONAL_ARGS $MODEL_NAME
# huggingface-cli download 

