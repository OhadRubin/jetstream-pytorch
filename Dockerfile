FROM us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pytorch-server:v0.2.4

# Set up the repository at build time instead of runtime
WORKDIR /jetstream-pytorch
RUN git config --global --add safe.directory /jetstream-pytorch && \
    git remote remove origin && \
    git remote add origin https://github.com/OhadRubin/jetstream-pytorch.git && \
    git fetch && \
    git checkout main && \
    git branch --set-upstream-to=origin/main main && \
    git pull

# Keep the original entrypoint

