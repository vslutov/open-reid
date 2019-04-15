#!/usr/bin/env zsh

source ~/.zshrc

if [ -z "$(ls ~/data)" ]
then
  cd ~/data
  tar xf ~/work/data.tar
  cd ~/work
fi

$@
