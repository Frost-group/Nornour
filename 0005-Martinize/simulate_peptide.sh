#!/bin/bash
#
#iterate through and generate some MD ^)

for peptide in ` cat peptides.dat `
do
    echo "fab ${peptide}; save ${peptide}.pdb" | pymol -c -p

#    gmx pdb2gmx -ignh  -f ${peptide}.pdb
#    gmx solvate -box 5 5 5 -cp conf.gro -p topol.top
#    gmx grompp -f emin-charmm.mdp -c out.gro

# Hold my beer
# Simulate protein MD with XTB (empirical tight-binding) model; uses Langevin
# dynamics and an implicit water model
 xtb --gfn 2 --md ${peptide}.pdb

done

