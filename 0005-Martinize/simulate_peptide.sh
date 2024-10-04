#!/bin/bash
#
#iterate through and generate some MD ^)

for peptide in ` cat peptides.dat `
do
    echo "fab ${peptide}; save ${peptide}.pdb" | pymol -c -p

# vaguely following https://cgmartini.nl/docs/tutorials/Martini3/ProteinsI/index.html

# Martinize2 vermouth - 'pip install vermouth'
martinize2 -f ${peptide}.pdb -o ${peptide}.top -x ${peptide}_cg.pdb

# vacuum energy minim
gmx editconf -f ${peptide}_cg.pdb -d 1.0 -bt dodecahedron -o ${peptide}_cg.gro
gmx grompp -maxwarn 1 -p ${peptide}.top -f minimization.mdp -c ${peptide}_cg.gro -o minimization-vac.tpr  -r ${peptide}_cg.gro
gmx mdrun -s minimization-vac.tpr -deffnm minimization-vac -v

gmx solvate -cp minimization-vac.gro -cs water.gro -radius 0.21 -o solvated.gro

waterbeads=` grep -c W solvated.gro `
echo "#include \"martini_v3.0.0_solvents_v1.itp\"" > ${peptide}-solvated.top
# the above order isn't correct, I had to hand edit it so the solvents were
# after the main include
cat ${peptide}.top >> ${peptide}-solvated.top
echo "W ${waterbeads}" >> ${peptide}-solvated.top

gmx grompp -p ${peptide}-solvated.top -c solvated.gro -f minimization.mdp -o minimization.tpr -r solvated.gro


# atomistic approach
#    gmx pdb2gmx -ignh  -f ${peptide}.pdb
#    gmx solvate -box 5 5 5 -cp conf.gro -p topol.top
#    gmx grompp -f emin-charmm.mdp -c out.gro

# Hold my beer
# Simulate protein MD with XTB (empirical tight-binding) model; uses Langevin
# dynamics and an implicit water model
# xtb --gfn 2 --md ${peptide}.pdb

done

