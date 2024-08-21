
# Function to generate an mdp file with given parameters
function generate_mdp {

# generate_mdp step1.mdp steep  0.001 1500 300 1.0 nvt null
g
  local filename=$1
  
  local integrator=$2
  local dt=$3
  local nsteps=$4
  local temp=$5
  local press=$6
  local thermostat=$7
  local barostat=$8

  cat <<EOF > "$filename"
integrator = $integrator 
dt = $dt
nsteps = $nsteps

; Temperature coupling
tau_t = 1.0
ref_t = $temp

; Pressure coupling
pcoupl = $barostat
ref_p =  $press 
compressibility = 4.5e-5

; Thermostat
tcoupl = $thermostat
tc-grps = system 

; Finally, most input parameters are set, for the production step, following
; the suggestions from De Jong et al. (de Jong, Baoukina, Ingólfsson, & Marrink,
; 2016) with a few notable changes. In a recent study (Kim, Fabian, & Hummer,
; 2023), it was shown that infrequent neighbor list updates can cause
; deformations in large membranes that are simulated with a semi-isotropic
; barostat as a result of missed non-bonded interactions. To avoid this artifact,
; it is suggested to override the automated Verlet buffer setting and to set the
; cut-off distance for the short-range neighbor list to 1.35 nm.

cutoff-scheme = Verlet 
nstlist = 20 
pbc = xyz 
verlet-buffer-tolerance = -1 
nsttcouple = 20 
nstpcouple = 20 
rlist = 1.35

; In addition, the cutoff distance for Coulomb and Lennard-Jones potentials was
; set to 1.1 nm, with a Potential-shift modifier to shift both potentials to zero
; at the cutoff. The relative dielectric constant is set to 15 as suggested in de
; Jong et al.

coulombtype = Reaction-Field 
coulomb-modifier = Potential-shift 
rcoulomb-switch = 0 
rcoulomb = 1.1 
epsilon-r = 15 
epsilon-rf = 0 
vdw-type = Cut-off 
vdw-modifier = Potential-shift 
rvdw-switch = 0 
rvdw = 1.1

; Output
EOF
}

# Generate mdp files for each step
generate_mdp step1.mdp steep  0.001 1500 300 1.0 no no
gmx grompp -c symmetric-bilayer.gro -p symmetric-bilayer.top -f step1.mdp
gmx mdrun 

#             filename  integrator timestep temp  
generate_mdp step2.mdp md 0.001    5000 300 1.0 berendsen berendsen
# maxwarn 2 to avoid grouching about the Berendsen thermostat
gmx grompp -maxwarn 2 -c confout.gro -p symmetric-bilayer.top -f step2.mdp
gmx mdrun 

exit

generate_mdp -maxwarn 2 step3.mdp md 0.005   20000 300 1.0 berendsen berendsen
generate_mdp -maxwarn 2 step4.mdp md 0.010   40000 300 1.0 berendsen berendsen
generate_mdp step5.mdp md 0.020  100000 300 1.0 v-rescale c-rescale
generate_mdp step6.mdp md 0.020 1000000 300 1.0 v-rescale c-rescale





