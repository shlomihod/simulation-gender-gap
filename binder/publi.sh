#!/bin/bash

set -e
set -x

nbinteract --execute simulation.ipynb

git checkout gh-pages

mv simulation.html simulation.html
git add simulation.html
git commit -m "automatic publish"
git push

git checkout master
