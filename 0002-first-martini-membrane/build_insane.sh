insane -x 20 -y 20 -z 10 -l DPPC:50 -l DOPC:50  -sol W -salt 0.15 -rand 0.095 -solr 0.5 -o symmetric-bilayer.gro -p symmetric-bilayer.top

# suggests Na+ and Cl- as names; but Martini 3 calls these Na and Cl
# It seems that GROMACS is clever enough to follow the naming
#
# So probably need to delete the - and + from the symmetric-bilayer.top as
# a post-process step

# At this point in time, I did this by hand-editing the symmetric-bilayer.top
# file

