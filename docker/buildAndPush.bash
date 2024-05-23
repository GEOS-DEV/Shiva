#!/bin/bash



images="ubuntu:22.04 ubuntu-cuda11:22.04-cuda11.8 ubi:8.9 ubi-cuda11:8.8-cuda11.8  ubi-cuda12:8.9-cuda12.4.1"

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


#  docker build ${dir} -t ${repo}
#  docker push ${repo}
done
