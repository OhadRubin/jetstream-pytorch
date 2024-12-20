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
RUN python3 -m pip install watchfiles fire
RUN python3 -m pip install jax==0.4.34 jaxlib==0.4.34 libtpu-nightly==0.1.dev20241008+nightly -f https://storage.googleapis.com/libtpu-releases/index.html

# Keep the original entrypoint


