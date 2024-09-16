# Clark, S., Jowitt, T.A., Harris, L.K., Knight, C.G., Dobson, C.B., 2021. The
# lexicon of antimicrobial peptides: a complete set of arginine and tryptophan
# sequences. Commun Biol 4, 1–14. https://doi.org/10.1038/s42003-021-02137-7

# Computer readable data on Figshare:
# Clark, Sam (2021). The Lexicon of Antimicrobial Peptides: a Complete Set of
# Arginine and Tryptophan Sequences. figshare. Collection.
# https://doi.org/10.6084/m9.figshare.c.5104931.v1

# Particularly, the raw data CSV: 

wget "https://figshare.com/ndownloader/files/26448101" 

cat 26448101 | cut -f1 -d\,  | grep -v sequence > RW_lexicon.dat

