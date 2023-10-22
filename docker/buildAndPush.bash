#!/bin/bash



images="ubi8:ubi8.8-854"
#  ubi8-cuda11:ubi8.8-cuda11.8.0  ubi8-cuda12:ubi8.8-cuda12.2.2

for image in $images; do
  imagePart=(${image//:/ })
  dir=${imagePart[0]}
  tag=${imagePart[1]}

  
  repo=geosx/ubi8:${tag}

  echo dir=${dir}
  echo repo=${repo}


docker build ${dir} -t ${repo}
docker push ${repo}
done


