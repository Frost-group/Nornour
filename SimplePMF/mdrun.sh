#!/bin/bash
for i in {0..99}; do
    gmx mdrun -deffnm umbrella${i}
done
