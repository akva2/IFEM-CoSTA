#!/bin/bash

function clone_ifem {
  # Clone IFEM
  if ! test -d ${WORKSPACE}/deps/IFEM
  then
    pushd .
    mkdir -p $WORKSPACE/deps/IFEM
    cd $WORKSPACE/deps/IFEM
    git init .
    git remote add origin https://github.com/OPM/IFEM
    git fetch --depth 1 origin $IFEM_REVISION:branch_to_build
    test $? -eq 0 || exit 1
    git checkout branch_to_build
    popd
  fi
}


# Upstreams and revisions
declare -a upstreams
upstreams=(IFEM-AdvectionDiffusion
           IFEM-Darcy)

declare -A upstreamRev
upstreamRev[IFEM-AdvectionDiffusion]=master
upstreamRev[IFEM-Darcy]=master

IFEM_REVISION=master
if grep -qi "ifem=" <<< $ghprbCommentBody
then
  IFEM_REVISION=pull/`echo $ghprbCommentBody | sed -r 's/.*ifem=([0-9]+).*/\1/g'`/merge
fi

clone_ifem

source $WORKSPACE/deps/IFEM/jenkins/build-ifem-module.sh

parseRevisions
printHeader IFEM-CoSTA

build_module_and_upstreams IFEM-CoSTA

test $? -eq 0 || exit 1

# No downstreams
