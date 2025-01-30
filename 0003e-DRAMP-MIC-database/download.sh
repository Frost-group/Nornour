# The DRAMP database offers perfectly formatted downloads of the data
# http://dramp.cpu-bioinfor.org/downloads/
#
# *Citation*:
# Shi G, Kang X, Dong F, Liu Y, Zhu N, Hu Y, Xu H, Lao X, Zheng H. DRAMP 3.0:
# an enhanced comprehensive data repository of antimicrobial peptides. Nucleic
# Acids Res. 2022 Jan 7;50(D1):D488-D496. PMID: 34390348

# wget "http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antibacterial_amps.txt" -O Antibacterial_amps.txt

# The original dataset was cleaned by only taking the AMPs with MIC values corresponding to p.aeruginosa bacteria
# The following rules were applied to sort the dataset:
# 1. Only peptides with canonical AA were kept
# 2. When multiple values of MIC where given, the average value was taken
# 3. When an uncertainty on the MIC value was given, it was removed to simplify the sorting process
# 4. All the MIC values were converted to µM

# This gives a dataset of 728 AMP's with MIC values against p.aeruginosa

