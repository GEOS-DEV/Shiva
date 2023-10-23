#!/bin/bash



images="ubuntu22:22.04 ubuntu22-cuda11:ubuntu22.04-cuda11.8.0 ubi8:ubi8.8-854 ubi8-cuda11:ubi8.8-cuda11.8.0  ubi8-cuda12:ubi8.8-cuda12.2.2"

for image in $images; do
  imagePart=(${image//:/ })
  dir=${imagePart[0]}
  tag=${imagePart[1]}

  dirPart=(${dir//-/ })
  dirRoot=${dirPart[0]}
  
  repo=geosx/${dirRoot}:${tag}

  echo 
  echo dir=${dir}
  echo repo=${repo}
  echo 


#docker build ${dir} -t ${repo}
#docker push ${repo}
done
