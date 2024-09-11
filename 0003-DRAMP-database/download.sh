# The DRAMP database offers perfectly formatted downloads of the data
# http://dramp.cpu-bioinfor.org/downloads/
#
# *Citation*:
# Shi G, Kang X, Dong F, Liu Y, Zhu N, Hu Y, Xu H, Lao X, Zheng H. DRAMP 3.0:
# an enhanced comprehensive data repository of antimicrobial peptides. Nucleic
# Acids Res. 2022 Jan 7;50(D1):D488-D496. PMID: 34390348

# (づ ᴗ _ ᴗ) づ ♡ - I love a good simple URL download
# wget "http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antibacterial_amps.txt" -O Antibacterial_amps.txt

cat Antibacterial_amps.txt | grep "^DRAMP" | awk '{print $2}' > Antibacterial_amps_sequences.txt

# Nb: Data still quite unclean! '24..spacerO()hxwlUgimvfJyqtZ' characters all
# turning up

# OK, this is now ready to use with a LSTM etc.; such as this super slick
# Javascript interface:
#    https://cs.stanford.edu/people/karpathy/recurrentjs/

