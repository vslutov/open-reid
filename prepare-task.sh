#!/usr/bin/env bash

get_script_dir () {
    SOURCE=$(readlink -f "${BASH_SOURCE[0]}")
    echo $( cd -P "$( dirname "$SOURCE" )" && pwd )
}

USERNAME=$(id -un)
WORK_FOLDER=$(get_script_dir)

WORK_BASENAME=$(basename "$WORK_FOLDER")
WORK_HASH=$(echo "$WORK_FOLDER" | md5sum - | head -c 6)
CONTAINER_NAME="${USERNAME}_deeplearning_${WORK_BASENAME}_${WORK_HASH}"
VOLUME_NAME="${CONTAINER_NAME}_data"
IMAGE="hpc.gml-team.ru:5000/${USERNAME}/open-reid"

NB_USER="user"

echo -n "Build docker image..."
docker build \
       --build-arg "USERNAME=${USERNAME}" \
       -t "$IMAGE" \
       "${WORK_FOLDER}" \
       >/dev/null
docker push "$IMAGE" >/dev/null
echo "Done"

srun --gres=gpu:1080ti:1 --mem 32768 -n1 --exclusive --partition=vision --nodelist=gml-gpu-01 \
     "${WORK_FOLDER}/run-task.sh" $@
