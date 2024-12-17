#!/bin/bash
#
#iterate through and generate some MD ^)

for peptide in ` cat peptides.dat `
do
# B and Z add terminals
    echo "fab B${peptide}Z; save ${peptide}.pdb" | pymol -c -p

# Hold my beer
# Simulate protein MD with XTB (empirical tight-binding) model; uses Langevin
# dynamics and an implicit water model
# xtb --gfn 2 --md ${peptide}.pdb
 
rm mdrestart xtbmdok gfnff_topo xtbtopo.mol

cat > md.input << EOF
\$md
   temp=310.0      # temperature (K)
   time=1.0        # simulation length (ps)
   step=2.0        # timestep (fs)
   dump=100.0      # Save coordinates every ... (fs)
   shake=1         # SHAKE on H-bonds only
   nvt=true        # NVT ensemble
   velo=310.0      # Initialize velocities at 310K
\$end

\$solvation
   solvent=water
   alpb=true
\$end
EOF

xtb --silent --gfn 2 --omd -I md.input ${peptide}.pdb

# obabel -ixyz xtb.trj -O last_frame.xyz -l1 #not the last frame?

cat > metadynamics.input << EOF
\$md
   restart=true

   temp=310.0
   time=1.0        # simulation length (ps)
   step=4.0        # timestep (fs)
   dump=50.0       # Save coordinates every ... (fs)
   shake=2         # SHAKE on all non-metal bonds
   hmass=4.0
   nvt=true        # NVT ensemble
   velo=310.0      # Initialize velocities at 310K
\$end

\$metadyn
   save=1.0        # Save Gaussian potentials every ... (ps)
   kpush=0.001     # Pushing constant ki/N
   alp=1.0         # Width of Gaussian potential
\$end

\$wall
   potential=logfermi
   sphere: auto, all   # Prevent decomposition
\$end

\$solvation
   solvent=water
   alpb=true
\$end
EOF

xtb --gfn 2 --md --silent -I metadynamics.input ${peptide}.pdb

#rm mdrestart xtbmdok gfnff_topo xtbtopo.mol
# xtb --silent --gfn 2  -I control.input 
 # --gfn 2
done

