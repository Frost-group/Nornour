
systemname=""

# Energy minimisation using the steepest descent algorithm - 1500 steps

gmx grompp -c ${systemname}.gro -r ${systemname}.gro -f minimisation.mdp -p ${systemname}.top -o em.tpr -maxwarn 5
gmx mdrun -s em.tpr -deffnm em 

# Time step: 1 fs , Total time: 5 ps , Thermostat: Berendsen , Barostat: Berendsen (τP = 10 ps)

gmx grompp -c em.gro -r em.gro -f equilibration1.mdp -p ${systemname}.top -o eq1.tpr -maxwarn 5 -n index.ndx
gmx mdrun -s eq1.tpr -deffnm eq1 

# Time step: 5 fs , Total time: 100 ps , Thermostat: Berendsen , Barostat: Berendsen (τP = 3 ps)

gmx grompp -c eq1.gro -r em.gro -f equilibration2.mdp -p ${systemname}.top -o eq2.tpr -maxwarn 5 -n index.ndx
gmx mdrun -s eq2.tpr -deffnm eq2  

# Time step: 10 fs , Total time: 400 ps , Thermostat: Berendsen , Barostat: Berendsen (τP = 3 ps)

gmx grompp -c eq2.gro -r em.gro -f equilibration3.mdp -p ${systemname}.top -o eq3.tpr -maxwarn 5 -n index.ndx
gmx mdrun -s eq3.tpr -deffnm eq3  

# Time step: 20 fs , Total time: 2 ns , Thermostat: v-rescale , Barostat: c-rescale (τP = 4 ps) 

gmx grompp -c eq3.gro -r em.gro -f preproduction.mdp -p ${systemname}.top -o preproduction.tpr -maxwarn 5 -n index.ndx 
gmx mdrun -s preproduction.tpr -deffnm preproduction  

# Time step: 20 fs , Total time: 2 µs , Thermostat: v-rescale , Barostat: c-rescale (τP = 4 ps) - production run will vary in timescale depending on lipid mixture 

gmx grompp -c preproduction.gro -r em.gro -f production.mdp -p ${systemname}.top -o production.tpr -maxwarn 5 -n index.ndx
gmx mdrun -s production.tpr -deffnm production 

# All commands and .mdp files have been downloaded/modified from https://doi.org/10.1016/bs.mie.2024.03.010 and https://bbs.llnl.gov/building_membranes_data.html - refer to for further information