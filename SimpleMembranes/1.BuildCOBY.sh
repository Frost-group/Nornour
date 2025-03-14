pip install COBY

# DOPC-CHOL membrane 
python -m COBY -box 20 20 20 -membrane lipid:DOPC:7 lipid:CHOL:3 -solvation default -sn DOPC_CHOL

# DOPE-DOPG membrane
python -m COBY -box 20 20 20 -membrane lipid:DOPE:7 lipid:DOPG:3 -solvation default -sn DOPE_DOPG

# General command
# python -m COBY -box x y x -membrane lipid:lipidtype:lipidratio -solvation solv:solventype pos:posion neg:negion -sn systemname

# Default box type is rectangular but can set hexagonal, skewed hexagonal and rhombic dodecahedron or manually designate the unit cell
# Default solvation corresponds to 0.15 M NaCl + any additional ions to counter charged lipid heads
# By default the command will output a .gro , .pdb and .top file 

# For additional commands and extra information refer to https://github.com/MikkelDA/COBY
# Specifically https://doi.org/10.1101/2024.07.23.604601 , https://github.com/MikkelDA/COBY/blob/master/COBY_Documentation.pdf and https://github.com/MikkelDA/COBY/blob/master/COBY_CHEAT_SHEET.pdf
