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

docker pull "$IMAGE" >/dev/null
docker volume create "${VOLUME_NAME}" >/dev/null

docker run \
       --runtime=nvidia \
       --shm-size 8G \
       --rm \
       -i \
       "--hostname=${WORK_BASENAME}" \
       "--volume=${WORK_FOLDER}:/home/${NB_USER}/work" \
       "--volume=${VOLUME_NAME}:/home/${NB_USER}/data" \
       -e CUDA_VISIBLE_DEVICES \
       "$IMAGE" \
       "/home/${NB_USER}/work/prepare-data.sh" \
       $@
