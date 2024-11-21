#!/bin/bash -x

# Function to prepare initial peptide structure
prepare_peptide() {
    # Generate peptide - using FASTA sequence input into CHAI fold web interface
    # (https://www.chaidiscovery.com/) - could set up on Linux - returns .cif file
    # which is manually converted to a .pdb using pymol. As I'm uploading this a new
    # *genuinely* open source code has come out which I probably will start to use -
    # Boltz-1 from MIT - https://github.com/jwohlwend/boltz?tab=readme-ov-file

    # Martnize2 uses martini3001 as a default, outputs .top .pdb and molecule_0.itp 
    echo "Preparing peptide structure..."
    martinize2 -f peptide.pdb -o peptide.top -x peptide_cg.pdb
}

# Function to insert peptide into membrane
insert_peptide() {
    # Insert peptide into equilibrated membrane box with at least 0.5 vDw radius
    # otherwise have infinite potential in the system due to atomic overlap,
    # insertpep.dat specifies coordinates for insertion (x,y,z) and specifying to
    # replace water -W Command output specifies how many water molecules have been
    # replaced, currently manually changing existing membrane.top file, subtracting
    # waters and adding [molecule_0 1] as well as '#include "molecule_0.itp"'
    echo "Inserting peptide into membrane..."
    gmx insert-molecules -f membrane.gro -ci peptide_cg.pdb -ip insertpep.dat -o system.gro -replace W -radius 0.5
}

# Function to neutralize system
neutralize_system() {
    # Generating .tpr file to use in next genion step
    echo "Neutralizing system..."
    gmx grompp -c system.gro -r system.gro -f minim.mdp -p topol.top -o em.tpr -maxwarn 5

    # genion neutralises system charge, equilibrated membrane is already neutral at
    # around 0.15 M NaCl, this will add in Cl ions to neutralise any positive amino
    # acid residues and automatically updates topology file, function asks for
    # continuous solvent system - 4 specifies to replace water (W) with ions
    echo "4" | gmx genion -s em.tpr -p topol.top -o systemion.gro -neutral
}

# Function to create index groups
create_index_groups() {
    # Make index file specifying which atoms belong to the Solvent, Membrane and
    # Protein as will specify groups in later gmx runs, gmx genion specifies added
    # Cl atoms separately to existing ions so ensure they're included in solvent
    echo "Creating index groups..."
    gmx make_ndx -f systemion.gro -o index.ndx << EOF
r W | r ION | r CL
name 1 Solvent
r DOPC | r CHOL
name 2 Membrane
!1 & !2
name 3 Protein
q
EOF
}

# Function for energy minimization
run_minimization() {
    # At this point run steep descent optimisation to fill in vacuum left by gmx
    # insert-molecules
    echo "Running energy minimization..."
    gmx grompp -c systemion.gro -r systemion.gro -f minim.mdp -p topol.top -o em.tpr -maxwarn 5
    gmx mdrun -s em.tpr -deffnm em
}

# Function for equilibration
run_equilibration() {
    # Next run a simple equilibration with Berendsen parameters. Have to do this as
    # the pull code will not work otherwise due to poor pressure scaling - would be
    # possible at this point to carry out more equilibration runs but would need to
    # restrict protein to stop large changes in initial protein position. For now
    # this gives sensible results.  Time step: 1 fs , Total time: 5 ps , Thermostat:
    # Berendsen , Barostat: Berendsen (τP = 10 ps) 
    echo "Running equilibration..."
    gmx grompp -c em.gro -r em.gro -f equilibration1.mdp -p topol.top -o eq1.tpr -maxwarn 5 -n index.ndx
    gmx mdrun -s eq1.tpr -deffnm eq1
}

# Function for production run
run_production() {
    # Pull run, protein is pulled 10 nm (half of box) over the course of 5 ns with
    # rescale parameters. Compressed trajectory is output with 200 points along
    # coordinate axis. First group defined (pull_group1_name) is always the
    # reference, in this case the membrane. Crucial parameter to define is the
    # pull_coord1_geometry - 'distance' will not allow to pull through the membrane
    # as cannot generate a negative distance. Can use 'direction' but may break
    # apart membrane due to long distance interactions between lipids, currently
    # using 'cylinder' which considers a radius around the pulled peptide so only
    # lipids in vicinity of peptide interact with it.  Time step: 20 fs , Total
    # time: 5 ns , Thermostat: v-rescale , Barostat: c-rescale (τP = 4 ps)
    echo "Running production MD..."
    gmx grompp -c eq1.gro -r systemion.gro -f production.mdp -p topol.top -o production.tpr -maxwarn 5 -n index.ndx
    gmx mdrun -s production.tpr -deffnm production
}

# Function for analysis
run_analysis() {
    # Python script from an online tutorial
    # (http://www.mdtutorials.com/gmx/umbrella/index.html) - will output the 200
    # snapshotted frames from pull run (conf 1-200) and also output a distance file
    # specifying how far the peptide travelled, giving its location in each frame
    # along the axis it was pulled through.
    echo "Running analysis..."
    bash distance_script.sh

    # List of gmx grompp commands to generate 100 tpr files for umbrella sampling
    # using 'production_umbrella.mdp' - file the same as 'production.mdp' used for
    # pulling with the exception that 'pull_coord1_rate = 0.0' - uses every odd conf
    # frame generated from previous step.  Time step: 20 fs , Total time: 5 ns ,
    # Thermostat: v-rescale , Barostat: c-rescale (τP = 4 ps)
    bash grompp_commands.sh
}

# Function for HPC instructions and final analysis
print_final_instructions() {
    # Generally from this point, transfer .tpr files from local machine to hpc. Then
    # queue mdrun.sh file - list of gmx mdrun commands for all umbrella.tpr files.
    # Once complete move back to local machine for analysis. 
    echo "Transfer files to HPC:"
    echo "scp umbrella* @login.hx1.hpc.ic.ac.uk:/gpfs/home/"
    echo "qsub -q hx hx1.pbs"
    echo "(bash mdrun.sh)"
    echo "scp -r @login.hx1.hpc.ic.ac.uk:/gpfs/home//xvgfiles /Users/home"

    # gmx wham takes input of .dat files - just a list of umbrella.tpr/.xvg files
    # 0-99 - performs weighted histogram analysis and outputs hist.xvg and pmf.xvg
    # files - specified units of KCal - can then plot in terminal using xmgrace
    # (brew install grace)
    echo -e "\nAfter HPC runs, for final analysis:"
    echo "gmx wham -it tpr-files.dat -if pullf-files.dat -o pmf.xvg -hist hist.xvg -unit kCal"
    echo "xmgrace -nxy hist.xvg"
    echo "xmgrace -nxy pmf.xvg"

    # currently trying to converge energy as the peptide is pulled out of the
    # membrane, for this I'm testing out different sample times for the umbrella
    # runs as well as pulling the peptide slower and adding more frames 
}

# Main execution flow
main() {
    prepare_peptide
    insert_peptide
    neutralize_system
    create_index_groups
    run_minimization
    run_equilibration
    run_production
    run_analysis
    print_final_instructions
}

# Execute main function
main
