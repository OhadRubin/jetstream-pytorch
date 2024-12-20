FROM us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pytorch-server:v0.2.4

# Set up the repository at build time instead of runtime
WORKDIR /jetstream-pytorch
RUN git config --global --add safe.directory /jetstream-pytorch && \
    git fetch && \
    git checkout bb174b62aad27a90f71ddea6d5fa0312e064bc50

# Keep the original entrypoint

git remote add origin https://github.com/OhadRubin/jetstream-pytorch.git