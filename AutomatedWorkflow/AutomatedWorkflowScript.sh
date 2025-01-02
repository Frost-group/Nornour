#!/bin/bash -x

# Fairly fragile script, need to rewrite some functions and clean it up but for the most part should work. 
# Need to rewrite to make compatible with ARCHER (add srun + distribute tasks across CPUs).

# Sub for desired sequence - can generate .pdb with boltz-1 once get that running locally
peptide="RWRWWWW"

# Currently sub for either DOPC_CHOL (human) or DOPE_DOPG (bacterial)
membrane="DOPC_CHOL"

# conda install conda-forge::gromacs
# conda install anaconda::pip
# pip install vermouth

# Function takes atomistic structure and generates martini model, renaming files to keep consistency throughout workflow.
prepare_peptide() {

    martinize2 -f ${peptide}.pdb -o ${peptide}_${membrane}.top -x ${peptide}_${membrane}_cg.pdb
    
    if [ -f molecule_0.itp ]; then
        sed "s/molecule_0/${peptide}/g" molecule_0.itp > ${peptide}.itp
        rm molecule_0.itp
    fi

    cp ${peptide}_${membrane}.top ${peptide}_${membrane}.top.tmp
    sed "s/molecule_0.itp/${peptide}.itp/g" ${peptide}_${membrane}.top.tmp | 
    sed "s/molecule_0/${peptide}/g" > ${peptide}_${membrane}.top
    rm ${peptide}_${membrane}.top.tmp
}

# Function inserts martini peptide into membrane model, merging topology files and updating water count.
# insertpep.dat specifies coordinates for insertion (x,y,z).
insert_peptide() {

    gmx insert-molecules -f ${membrane}.gro -ci ${peptide}_${membrane}_cg.pdb -ip insertpep.dat -o ${peptide}_${membrane}.gro -replace W -radius 0.5

   cat ${membrane}.top > ${peptide}_${membrane}_temp.top
   
   {
       echo "#include \"martini_v3.0.0.itp\""
       echo "#include \"martini_v3.0.0_phospholipids_v1.itp\""
       echo "#include \"martini_v3.0_sterols_v1.0.itp\""
       echo "#include \"martini_v3.0.0_solvents_v1.itp\""
       echo "#include \"martini_v3.0.0_ions_v1.itp\""
       echo "#include \"${peptide}.itp\""
       echo "[ system ]"
       echo "; name"
       echo "${membrane}.top"
       echo "[ molecules ]"
       echo "; name number"
       grep -A10 "\[ molecules \]" ${peptide}_${membrane}_temp.top | tail -n +3
       echo "${peptide}    1"
   } > ${peptide}_${membrane}.top
   
   rm ${peptide}_${membrane}_temp.top

	water_count=$(awk '$1 ~ /W$/ {count++} END {print count}' "${peptide}_${membrane}.gro")
	awk -v count=$water_count '/^W /{printf "W    %d\n",count; next} {print}' "${peptide}_${membrane}.top" > tmp && mv tmp "${peptide}_${membrane}.top"
}

# Function neutralises system (accounts for charge of peptide).
neutralise_system() {

    gmx grompp -c ${peptide}_${membrane}.gro -r ${peptide}_${membrane}.gro -f minim.mdp -p ${peptide}_${membrane}.top -o ${peptide}_${membrane}.tpr -maxwarn 5

    echo "4" | gmx genion -s ${peptide}_${membrane}.tpr -p ${peptide}_${membrane}.top -o ${peptide}_${membrane}_ion.gro -neutral
}

# Function makes index file specifying which atoms belong to the solvent, membrane and protein.
create_index_groups() {

    echo "keep 0" > ${peptide}_${membrane}_index-selection.txt
    echo "r W | r ION | r CL" >> ${peptide}_${membrane}_index-selection.txt
    echo "name 1 Solvent" >> ${peptide}_${membrane}_index-selection.txt
    echo "r DOPC | r CHOL" >> ${peptide}_${membrane}_index-selection.txt
    echo "name 2 Membrane" >> ${peptide}_${membrane}_index-selection.txt
    echo '!1 & !2' >> ${peptide}_${membrane}_index-selection.txt
    echo "name 3 Protein" >> ${peptide}_${membrane}_index-selection.txt
    echo "q" >> ${peptide}_${membrane}_index-selection.txt

    gmx make_ndx -f ${peptide}_${membrane}_ion.gro -o ${peptide}_${membrane}.ndx < ${peptide}_${membrane}_index-selection.txt
}

# Steep descent optimisation to fill in vacuum left by gmx insert-molecules.
run_minimisation() {

    gmx grompp -c ${peptide}_${membrane}_ion.gro -r ${peptide}_${membrane}_ion.gro -f minim.mdp -p ${peptide}_${membrane}.top -o ${peptide}_${membrane}_em.tpr -maxwarn 5
    gmx mdrun -s ${peptide}_${membrane}_em.tpr -deffnm ${peptide}_${membrane}_em
}

# Simple equilibration - Time step: 1 fs , Total time: 5 ps , Thermostat: Berendsen , Barostat: Berendsen (τP = 10 ps).
run_equilibration() {

    gmx grompp -c ${peptide}_${membrane}_em.gro -r ${peptide}_${membrane}_em.gro -f equilibration.mdp -p ${peptide}_${membrane}.top -o ${peptide}_${membrane}_eq.tpr -maxwarn 5 -n ${peptide}_${membrane}.ndx
    gmx mdrun -s ${peptide}_${membrane}_eq.tpr -deffnm ${peptide}_${membrane}_eq
}

# Pulling peptide across membrane - Time step: 20 fs , Total time: 10.24 ns , Thermostat: v-rescale , Barostat: c-rescale (τP = 4 ps).
run_production() {

    gmx grompp -c ${peptide}_${membrane}_eq.gro -r ${peptide}_${membrane}_ion.gro -f production.mdp -p ${peptide}_${membrane}.top -o ${peptide}_${membrane}_production.tpr -maxwarn 5 -n ${peptide}_${membrane}.ndx
    gmx mdrun -s ${peptide}_${membrane}_production.tpr -deffnm ${peptide}_${membrane}_production -v
}

# Function for running umbrella sampling. Splits production run into frames and then samples a subset.
# Time step: 20 fs , Total time: 10.24 ns , Thermostat: v-rescale , Barostat: c-rescale (τP = 4 ps).
run_umbrella() {

    echo 0 | gmx trjconv -s ${peptide}_${membrane}_production.tpr -f ${peptide}_${membrane}_production.xtc -o ${peptide}_${membrane}_conf.gro -sep

    for (( i=0; i<257; i++ ))
    do
    	gmx distance -s ${peptide}_${membrane}_production.tpr -f ${peptide}_${membrane}_conf${i}.gro -n ${peptide}_${membrane}.ndx -select 'com of group "Protein" plus com of group "Membrane"'
    done
  
    for i in {0..127}; do
    	conf_num=$((2*i + 1))
    	gmx grompp -f production_umbrella.mdp -c ${peptide}_${membrane}_conf${conf_num}.gro -p ${peptide}_${membrane}.top -r ${peptide}_${membrane}_conf${conf_num}.gro -n ${peptide}_${membrane}.ndx -o ${peptide}_${membrane}_umbrella${i}.tpr -maxwarn 2
    done
}

# Function takes input of .dat files, performs weighted histogram analysis and outputs pmf. 
run_wham() {
 
    for i in {0..127}
    do
    	echo "${peptide}_${membrane}_umbrella${i}.tpr" >> ${peptide}_${membrane}_tpr-files.dat
    done

    for i in {0..127}
    do
    	echo "${peptide}_${membrane}_umbrella${i}_pullx.xvg" >> ${peptide}_${membrane}_pullx-files.dat
    done
   
    gmx wham -it ${peptide}_${membrane}_tpr-files.dat -ix ${peptide}_${membrane}_pullx-files.dat -o ${peptide}_${membrane}_pmf.xvg -hist ${peptide}_${membrane}_hist.xvg -unit kCal   
}

main() {
    prepare_peptide
    insert_peptide
    neutralise_system
    create_index_groups
    run_minimisation
    run_equilibration
    run_production
    run_umbrella
}

main
