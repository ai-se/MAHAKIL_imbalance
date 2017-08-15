#!/usr/bin/env bash

rm dump/*
rm out/*
rm err/*
rm log/*

foreach VAR ("tomcat" "synapse" "camel" "ant" "arc" "ivy" "velocity" "redaktor" "jedit")
  bsub -q standard -W 5000 -n 8 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share2/aagrawa8/miniconda2/bin/python2.7 mahakil.py _test "$VAR" > log/"$VAR".log
end