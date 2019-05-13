#!/bin/bash

set -e
set -x

nbinteract --spec shlomihod/simulation-gender-gap/master --execute simulation.ipynb

git checkout gh-pages

mv simulation.html index.html
git add index.html
git commit -m "automatic publish"
git push

git checkout master
