#!/bin/bash
for i in {0..99}; do
    conf_num=$((2*i + 1))
    gmx grompp -f production_umbrella.mdp -c conf${conf_num}.gro -p topol.top -r conf${conf_num}.gro -n index.ndx -o umbrella${i}.tpr -maxwarn 2
done
