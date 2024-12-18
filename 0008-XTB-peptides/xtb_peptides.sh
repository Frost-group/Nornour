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

mkdir -p ${peptide}
cd ${peptide}


# B and Z add terminals
   echo "fab B${peptide}Z; save ${peptide}.pdb" | pymol -c -p
   charge=` echo ${peptide} | grep -o '[RHK]' | wc -l ` # count number of cation residues

   echo "Peptide: ${peptide} Inferred charge: ${charge}"

for solvent in "water" "octanol" "woctanol"
do

mkdir -p md-${solvent}
cd md-${solvent}

rm mdrestart xtbmdok gfnff_topo xtbtopo.mol
# Vaguely following protocol in
# Quantum chemical molecular dynamics and metadynamics simulation of aluminium
# binding to amyloid-β and related peptides
# James A. Platts 
# Published:05 February 2020
# https://doi.org/10.1098/rsos.191562

cat > md.input << EOF
\$md
   temp=300.0      # temperature (K)
   time=10.0        # simulation length (ps)
   step=2.0        # timestep (fs)
   dump=100.0      # Save coordinates every ... (fs)
   shake=1         # SHAKE on H-bonds only
   nvt=true        # NVT ensemble
   velo=300.0      # Initialize velocities at ... (K)
\$end

\$solvation
   solvent=${solvent}
   alpb=true
\$end
EOF

xtb  --gfn 2 --omd -I md.input --chrg ${charge} ../${peptide}.pdb > xtb_md.log

cd ..

mkdir -p metadynamics-${solvent}
cd metadynamics-${solvent}
# bring restart file (MD velocities and positions)
cp -a ../md-${solvent}/mdrestart ./

cat > metadynamics.input << EOF
\$md
   restart=true

   temp=300.0
   time=100.0        # simulation length (ps)
   step=4.0        # timestep (fs)
   dump=50.0       # Save coordinates every ... (fs)
   shake=2         # SHAKE on all non-metal bonds
   hmass=4.0
   nvt=true        # NVT ensemble
   velo=300.0      # Initialize velocities at ... (K)
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
   solvent=${solvent}
   alpb=true
\$end
EOF

xtb --gfn 2 --md -I metadynamics.input --chrg ${charge} ../${peptide}.pdb > xtb_metadynamics.log

#rm mdrestart xtbmdok gfnff_topo xtbtopo.mol
# xtb --silent --gfn 2  -I control.input 
# --gfn 2

cd ..

done



cd ..

done

