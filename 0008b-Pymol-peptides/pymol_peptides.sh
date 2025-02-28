#!/bin/bash
#
#iterate through and generate some MD ^)
#
# Hold my beer
# Simulate protein MD with XTB (empirical tight-binding) model; uses Langevin
# dynamics and an implicit water model
# xtb --gfn 2 --md ${peptide}.pdb


for peptide in ` cat peptides.dat `
do
# B and Z add terminals
   echo "fab B${peptide}Z; save ${peptide}.pdb" | pymol -c -p
   charge=` echo ${peptide} | grep -o '[RHK]' | wc -l ` # count number of cation residues

   echo "Peptide: ${peptide} Inferred charge: ${charge}"
done

