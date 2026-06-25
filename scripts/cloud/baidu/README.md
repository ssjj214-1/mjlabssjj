# Baidu AIHC training

Use one Docker image for both Ultra GameYaw V12 and V13. Runtime outputs such as
`logs/`, `wandb/`, `.venv/`, and caches are excluded by the repository
`.dockerignore`, so historical training logs are not copied into the image.

## 1. Create CCR repository

In Baidu Cloud Console:

1. Open Container Registry CCR.
2. Create or select a namespace, for example `mjlabssjj`.
3. Create a repository, for example `mjlabssjj`.
4. Copy the CCR `docker login` command from the console and run it locally.

The final image address will look like:

```text
registry.baidubce.com/mjlabssjj/mjlabssjj:v12-v13
```

## 2. Build and push

Run from the repository root:

```bash
CCR_NAMESPACE=mjlabssjj CCR_REPOSITORY=mjlabssjj \
  scripts/cloud/baidu/build_and_push.sh
```

After the push finishes, refresh the AIHC CCR image selector. The image should
appear under the selected CCR namespace/repository.

## 3. Create AIHC jobs

Use the same image for both jobs. Mount persistent storage at `/workspace` so
checkpoints and TensorBoard logs survive container exit.

V12 job:

```bash
MJLAB_TASK=Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V12 \
  scripts/cloud/baidu/train_aihc.sh
```

V13 job:

```bash
MJLAB_TASK=Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V13 \
  scripts/cloud/baidu/train_aihc.sh
```

Useful optional environment variables:

```text
LOG_ROOT=/workspace/logs/rsl_rl
NUM_ENVS=4096
MAX_ITERATIONS=20000
GPU_IDS=all
```

Set the AIHC TensorBoard path to:

```text
/workspace/logs/rsl_rl
```
