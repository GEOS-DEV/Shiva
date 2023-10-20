#!/bin/bash



images="ubi8:ubi8.8"

for image in $images; do
  imagePart=(${image//:/ })
  dir=${imagePart[0]}
  tag=${imagePart[1]}
  repo=geosx/${imagePart[0]}:${tag}
  echo ${repo}

  docker build ${dir} -t ${tag}
  docker push ${repo}
done


