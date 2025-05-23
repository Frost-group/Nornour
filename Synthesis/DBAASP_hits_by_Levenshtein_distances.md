# Close hits in DBAASP by Levenshtein distances.

Levenshtein distances calculated by Julia code, with a weight of 1x for codon
substitution, and a 2x cost for codon deletion or addition. (Assuming this
would significantly change function.)

Code should be at: Levenshtein_distances_pluto.jl 

## Peptide : peptide_1 	Seq: IVFLKLPAGKKKIL
[Match distance: 7  	Seq: FLLKLGLGKKKLL](https://www.dbaasp.org/search?sequence.value=FLLKLGLGKKKLL)

[Match distance: 7  	Seq: GNLKKLLAVAKKIL](https://www.dbaasp.org/search?sequence.value=GNLKKLLAVAKKIL)

[Match distance: 7  	Seq: IDWKKLPDAAKQIL](https://www.dbaasp.org/search?sequence.value=IDWKKLPDAAKQIL)

[Match distance: 7  	Seq: INLKALAALAKKIL](https://www.dbaasp.org/search?sequence.value=INLKALAALAKKIL)

[Match distance: 7  	Seq: LNLKKLAAVAKKIL](https://www.dbaasp.org/search?sequence.value=LNLKKLAAVAKKIL)

[Match distance: 7  	Seq: LNLKKLLAVAKKIL](https://www.dbaasp.org/search?sequence.value=LNLKKLLAVAKKIL)

[Match distance: 8  	Seq: ANLKALLAVAKKIL](https://www.dbaasp.org/search?sequence.value=ANLKALLAVAKKIL)

[Match distance: 8  	Seq: GILGKLWEGVKSIL](https://www.dbaasp.org/search?sequence.value=GILGKLWEGVKSIL)

[Match distance: 8  	Seq: GNLKALAAVAKKIL](https://www.dbaasp.org/search?sequence.value=GNLKALAAVAKKIL)

[Match distance: 8  	Seq: GNLKALLAVAKKIL](https://www.dbaasp.org/search?sequence.value=GNLKALLAVAKKIL)

## Peptide : peptide_8 	Seq: IVLPKLKCLLIK
[Match distance: 6  	Seq: KLKKKLKKLLKK](https://www.dbaasp.org/search?sequence.value=KLKKKLKKLLKK)

[Match distance: 6  	Seq: KLLLKLKLKLLK](https://www.dbaasp.org/search?sequence.value=KLLLKLKLKLLK)

[Match distance: 7  	Seq: CKLKKLWKKLLK](https://www.dbaasp.org/search?sequence.value=CKLKKLWKKLLK)

[Match distance: 7  	Seq: CLLKKLLKKLLKK](https://www.dbaasp.org/search?sequence.value=CLLKKLLKKLLKK)

[Match distance: 7  	Seq: GLLKRIKTLLKK](https://www.dbaasp.org/search?sequence.value=GLLKRIKTLLKK)

[Match distance: 7  	Seq: KFFRKLKKLVKK](https://www.dbaasp.org/search?sequence.value=KFFRKLKKLVKK)

[Match distance: 7  	Seq: KKLLKLLLKLLK](https://www.dbaasp.org/search?sequence.value=KKLLKLLLKLLK)

[Match distance: 7  	Seq: KLKKLLKKLLKK](https://www.dbaasp.org/search?sequence.value=KLKKLLKKLLKK)

[Match distance: 7  	Seq: KLLGKWKGLLK](https://www.dbaasp.org/search?sequence.value=KLLGKWKGLLK)

[Match distance: 7  	Seq: KLLKWLKKLLK](https://www.dbaasp.org/search?sequence.value=KLLKWLKKLLK)

## Peptide : peptide_10 	Seq: IKFLSLKLGLSLKK
[Match distance: 6  	Seq: KKLLLGLGLLLKK](https://www.dbaasp.org/search?sequence.value=KKLLLGLGLLLKK)

[Match distance: 7  	Seq: KLKLKLKLKLKLKL](https://www.dbaasp.org/search?sequence.value=KLKLKLKLKLKLKL)

[Match distance: 7  	Seq: LKLKLKLKLKLKL](https://www.dbaasp.org/search?sequence.value=LKLKLKLKLKLKL)

[Match distance: 8  	Seq: FGKLLKLGKGLKG](https://www.dbaasp.org/search?sequence.value=FGKLLKLGKGLKG)

[Match distance: 8  	Seq: GKKLGLWLGLKKG](https://www.dbaasp.org/search?sequence.value=GKKLGLWLGLKKG)

[Match distance: 8  	Seq: GLFDILKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFDILKKLLKLLK)

[Match distance: 8  	Seq: GLFDLLKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFDLLKKLLKLLK)

[Match distance: 8  	Seq: GLFKLLKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFKLLKKLLKLLK)

[Match distance: 8  	Seq: GLKLLLKLGLKLL](https://www.dbaasp.org/search?sequence.value=GLKLLLKLGLKLL)

[Match distance: 8  	Seq: GLSLLLKLGLKLL](https://www.dbaasp.org/search?sequence.value=GLSLLLKLGLKLL)

## Peptide : peptide_11 	Seq: LILKPLKLLKCLKKL
[Match distance: 6  	Seq: LKKLLKLLKKLLKL](https://www.dbaasp.org/search?sequence.value=LKKLLKLLKKLLKL)

[Match distance: 6  	Seq: LKKLLKLLKKLPKL](https://www.dbaasp.org/search?sequence.value=LKKLLKLLKKLPKL)

[Match distance: 6  	Seq: LKKLLKLLKPLLKL](https://www.dbaasp.org/search?sequence.value=LKKLLKLLKPLLKL)

[Match distance: 6  	Seq: LKKPLKLGKKLLKL](https://www.dbaasp.org/search?sequence.value=LKKPLKLGKKLLKL)

[Match distance: 6  	Seq: LLCKKLKLKKLCLKKL](https://www.dbaasp.org/search?sequence.value=LLCKKLKLKKLCLKKL)

[Match distance: 6  	Seq: LLKKPLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=LLKKPLLKKLLKKL)

[Match distance: 7  	Seq: GILKKILKKILKKL](https://www.dbaasp.org/search?sequence.value=GILKKILKKILKKL)

[Match distance: 7  	Seq: GLLKKLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=GLLKKLLKKLLKKL)

[Match distance: 7  	Seq: KILKLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=KILKLLKKFLKKL)

[Match distance: 7  	Seq: KIPKPLKKFLKKL](https://www.dbaasp.org/search?sequence.value=KIPKPLKKFLKKL)

## Peptide : peptide_8 	Seq: IVLPKLKCLLIK
[Match distance: 6  	Seq: KLKKKLKKLLKK](https://www.dbaasp.org/search?sequence.value=KLKKKLKKLLKK)

[Match distance: 6  	Seq: KLLLKLKLKLLK](https://www.dbaasp.org/search?sequence.value=KLLLKLKLKLLK)

[Match distance: 7  	Seq: CKLKKLWKKLLK](https://www.dbaasp.org/search?sequence.value=CKLKKLWKKLLK)

[Match distance: 7  	Seq: CLLKKLLKKLLKK](https://www.dbaasp.org/search?sequence.value=CLLKKLLKKLLKK)

[Match distance: 7  	Seq: GLLKRIKTLLKK](https://www.dbaasp.org/search?sequence.value=GLLKRIKTLLKK)

[Match distance: 7  	Seq: KFFRKLKKLVKK](https://www.dbaasp.org/search?sequence.value=KFFRKLKKLVKK)

[Match distance: 7  	Seq: KKLLKLLLKLLK](https://www.dbaasp.org/search?sequence.value=KKLLKLLLKLLK)

[Match distance: 7  	Seq: KLKKLLKKLLKK](https://www.dbaasp.org/search?sequence.value=KLKKLLKKLLKK)

[Match distance: 7  	Seq: KLLGKWKGLLK](https://www.dbaasp.org/search?sequence.value=KLLGKWKGLLK)

[Match distance: 7  	Seq: KLLKWLKKLLK](https://www.dbaasp.org/search?sequence.value=KLLKWLKKLLK)

## Peptide : peptide_10 	Seq: IKFLSLKLGLSLKK
[Match distance: 6  	Seq: KKLLLGLGLLLKK](https://www.dbaasp.org/search?sequence.value=KKLLLGLGLLLKK)

[Match distance: 7  	Seq: KLKLKLKLKLKLKL](https://www.dbaasp.org/search?sequence.value=KLKLKLKLKLKLKL)

[Match distance: 7  	Seq: LKLKLKLKLKLKL](https://www.dbaasp.org/search?sequence.value=LKLKLKLKLKLKL)

[Match distance: 8  	Seq: FGKLLKLGKGLKG](https://www.dbaasp.org/search?sequence.value=FGKLLKLGKGLKG)

[Match distance: 8  	Seq: GKKLGLWLGLKKG](https://www.dbaasp.org/search?sequence.value=GKKLGLWLGLKKG)

[Match distance: 8  	Seq: GLFDILKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFDILKKLLKLLK)

[Match distance: 8  	Seq: GLFDLLKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFDLLKKLLKLLK)

[Match distance: 8  	Seq: GLFKLLKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFKLLKKLLKLLK)

[Match distance: 8  	Seq: GLKLLLKLGLKLL](https://www.dbaasp.org/search?sequence.value=GLKLLLKLGLKLL)

[Match distance: 8  	Seq: GLSLLLKLGLKLL](https://www.dbaasp.org/search?sequence.value=GLSLLLKLGLKLL)

## Peptide : peptide_11 	Seq: LILKPLKLLKCLKKL
[Match distance: 6  	Seq: LKKLLKLLKKLLKL](https://www.dbaasp.org/search?sequence.value=LKKLLKLLKKLLKL)

[Match distance: 6  	Seq: LKKLLKLLKKLPKL](https://www.dbaasp.org/search?sequence.value=LKKLLKLLKKLPKL)

[Match distance: 6  	Seq: LKKLLKLLKPLLKL](https://www.dbaasp.org/search?sequence.value=LKKLLKLLKPLLKL)

[Match distance: 6  	Seq: LKKPLKLGKKLLKL](https://www.dbaasp.org/search?sequence.value=LKKPLKLGKKLLKL)

[Match distance: 6  	Seq: LLCKKLKLKKLCLKKL](https://www.dbaasp.org/search?sequence.value=LLCKKLKLKKLCLKKL)

[Match distance: 6  	Seq: LLKKPLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=LLKKPLLKKLLKKL)

[Match distance: 7  	Seq: GILKKILKKILKKL](https://www.dbaasp.org/search?sequence.value=GILKKILKKILKKL)

[Match distance: 7  	Seq: GLLKKLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=GLLKKLLKKLLKKL)

[Match distance: 7  	Seq: KILKLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=KILKLLKKFLKKL)

[Match distance: 7  	Seq: KIPKPLKKFLKKL](https://www.dbaasp.org/search?sequence.value=KIPKPLKKFLKKL)

## Peptide : peptide_13 	Seq: ILFIGSLIKKRPIK
[Match distance: 8  	Seq: IKKILSKIKKLLK](https://www.dbaasp.org/search?sequence.value=IKKILSKIKKLLK)

[Match distance: 8  	Seq: IKKILSKIKKLWK](https://www.dbaasp.org/search?sequence.value=IKKILSKIKKLWK)

[Match distance: 8  	Seq: IKKILSKIKKWLK](https://www.dbaasp.org/search?sequence.value=IKKILSKIKKWLK)

[Match distance: 8  	Seq: IKKILSKIKKWWK](https://www.dbaasp.org/search?sequence.value=IKKILSKIKKWWK)

[Match distance: 8  	Seq: IKKIVSKIKKLLK](https://www.dbaasp.org/search?sequence.value=IKKIVSKIKKLLK)

[Match distance: 8  	Seq: IKKIVSKIKKVLK](https://www.dbaasp.org/search?sequence.value=IKKIVSKIKKVLK)

[Match distance: 8  	Seq: IKKIWSKIKKLLK](https://www.dbaasp.org/search?sequence.value=IKKIWSKIKKLLK)

[Match distance: 8  	Seq: IKKIWSKIKKLWK](https://www.dbaasp.org/search?sequence.value=IKKIWSKIKKLWK)

[Match distance: 8  	Seq: IKKIWSKIKKWLK](https://www.dbaasp.org/search?sequence.value=IKKIWSKIKKWLK)

[Match distance: 8  	Seq: IKKIWSKIKKWWK](https://www.dbaasp.org/search?sequence.value=IKKIWSKIKKWWK)

## Peptide : peptide_16 	Seq: LWFIVKKLASKVLP
[Match distance: 7  	Seq: LGALFKVASKVLP](https://www.dbaasp.org/search?sequence.value=LGALFKVASKVLP)

[Match distance: 8  	Seq: FFPIVKKLLKLLF](https://www.dbaasp.org/search?sequence.value=FFPIVKKLLKLLF)

[Match distance: 8  	Seq: FFPIVKKLLSGLF](https://www.dbaasp.org/search?sequence.value=FFPIVKKLLSGLF)

[Match distance: 8  	Seq: FIPIVKKLLSALF](https://www.dbaasp.org/search?sequence.value=FIPIVKKLLSALF)

[Match distance: 8  	Seq: FIPIVKKLLSGLF](https://www.dbaasp.org/search?sequence.value=FIPIVKKLLSGLF)

[Match distance: 8  	Seq: FLPIVAKLLSKLL](https://www.dbaasp.org/search?sequence.value=FLPIVAKLLSKLL)

[Match distance: 8  	Seq: FLPIVKKLLEKLL](https://www.dbaasp.org/search?sequence.value=FLPIVKKLLEKLL)

[Match distance: 8  	Seq: FLPIVKKLLKELL](https://www.dbaasp.org/search?sequence.value=FLPIVKKLLKELL)

[Match distance: 8  	Seq: FLPIVKKLLKGLF](https://www.dbaasp.org/search?sequence.value=FLPIVKKLLKGLF)

[Match distance: 8  	Seq: FLPIVKKLLKQLF](https://www.dbaasp.org/search?sequence.value=FLPIVKKLLKQLF)

## Peptide : peptide_17 	Seq: FRFPRIGIIILAVKK
[Match distance: 10  	Seq: FFPVIGRILNGIL](https://www.dbaasp.org/search?sequence.value=FFPVIGRILNGIL)

[Match distance: 10  	Seq: FIFPKKNIINSLFGR](https://www.dbaasp.org/search?sequence.value=FIFPKKNIINSLFGR)

[Match distance: 10  	Seq: FLTGLIGGLMKALGK](https://www.dbaasp.org/search?sequence.value=FLTGLIGGLMKALGK)

[Match distance: 10  	Seq: IRRIIRKIIHIIKK](https://www.dbaasp.org/search?sequence.value=IRRIIRKIIHIIKK)

[Match distance: 10  	Seq: LRRIIRKIIHIIKK](https://www.dbaasp.org/search?sequence.value=LRRIIRKIIHIIKK)

[Match distance: 11  	Seq: AIPFIFIFRLLRKG](https://www.dbaasp.org/search?sequence.value=AIPFIFIFRLLRKG)

[Match distance: 11  	Seq: AIPFIFIWRLLRKG](https://www.dbaasp.org/search?sequence.value=AIPFIFIWRLLRKG)

[Match distance: 11  	Seq: AIPFIWIFRLLRKG](https://www.dbaasp.org/search?sequence.value=AIPFIWIFRLLRKG)

[Match distance: 11  	Seq: AIPWIFIFRLLRKG](https://www.dbaasp.org/search?sequence.value=AIPWIFIFRLLRKG)

[Match distance: 11  	Seq: AIPWIWIWRLLRKG](https://www.dbaasp.org/search?sequence.value=AIPWIWIWRLLRKG)

## Peptide : peptide_97 	Seq: VFFWLLCKLKKKLL
[Match distance: 6  	Seq: FAKLLAKLAKKLL](https://www.dbaasp.org/search?sequence.value=FAKLLAKLAKKLL)

[Match distance: 6  	Seq: VAKLLAKLAKKLL](https://www.dbaasp.org/search?sequence.value=VAKLLAKLAKKLL)

[Match distance: 7  	Seq: FAKALAKLAKKLL](https://www.dbaasp.org/search?sequence.value=FAKALAKLAKKLL)

[Match distance: 7  	Seq: FAKLLAKALKKLL](https://www.dbaasp.org/search?sequence.value=FAKLLAKALKKLL)

[Match distance: 7  	Seq: FAKLLAKLAKKAL](https://www.dbaasp.org/search?sequence.value=FAKLLAKLAKKAL)

[Match distance: 7  	Seq: FAKLLAKLAKKIL](https://www.dbaasp.org/search?sequence.value=FAKLLAKLAKKIL)

[Match distance: 7  	Seq: FAKLLAKLAKKVL](https://www.dbaasp.org/search?sequence.value=FAKLLAKLAKKVL)

[Match distance: 7  	Seq: FFGSLLKLLPKLL](https://www.dbaasp.org/search?sequence.value=FFGSLLKLLPKLL)

[Match distance: 7  	Seq: GLFDLLKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFDLLKKLLKLLK)

[Match distance: 7  	Seq: GLFKLLKKLLKLLK](https://www.dbaasp.org/search?sequence.value=GLFKLLKKLLKLLK)

## Peptide : peptide_1181 	Seq: VLKCLCLKLKKKLL
[Match distance: 6  	Seq: GLKLLLKLGLKLL](https://www.dbaasp.org/search?sequence.value=GLKLLLKLGLKLL)

[Match distance: 6  	Seq: KLKKLLKKLKKLLK](https://www.dbaasp.org/search?sequence.value=KLKKLLKKLKKLLK)

[Match distance: 6  	Seq: KLKLKLKLKLKLK](https://www.dbaasp.org/search?sequence.value=KLKLKLKLKLKLK)

[Match distance: 6  	Seq: KLKLLLKLGLKLL](https://www.dbaasp.org/search?sequence.value=KLKLLLKLGLKLL)

[Match distance: 6  	Seq: LKLKKLCLCKLKKKLL](https://www.dbaasp.org/search?sequence.value=LKLKKLCLCKLKKKLL)

[Match distance: 6  	Seq: LLKALKKLLKKLL](https://www.dbaasp.org/search?sequence.value=LLKALKKLLKKLL)

[Match distance: 6  	Seq: LLKKLKCLCKLKKKLL](https://www.dbaasp.org/search?sequence.value=LLKKLKCLCKLKKKLL)

[Match distance: 6  	Seq: LLKLLLKLLLKLL](https://www.dbaasp.org/search?sequence.value=LLKLLLKLLLKLL)

[Match distance: 6  	Seq: VAKLLAKLAKKLL](https://www.dbaasp.org/search?sequence.value=VAKLLAKLAKKLL)

[Match distance: 7  	Seq: AAKKVLKLLKKLL](https://www.dbaasp.org/search?sequence.value=AAKKVLKLLKKLL)

## Peptide : peptide_1304 	Seq: WLLLLCLKKCLKKKL
[Match distance: 6  	Seq: LKCLLLLKKKCKKKL](https://www.dbaasp.org/search?sequence.value=LKCLLLLKKKCKKKL)

[Match distance: 6  	Seq: LLLKKLLKKCLKCKK](https://www.dbaasp.org/search?sequence.value=LLLKKLLKKCLKCKK)

[Match distance: 6  	Seq: WILLLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=WILLLLKKFLKKL)

[Match distance: 7  	Seq: ALLSLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=ALLSLLKKLLKKL)

[Match distance: 7  	Seq: GLLKKLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=GLLKKLLKKLLKKL)

[Match distance: 7  	Seq: GLLSLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=GLLSLLKKLLKKL)

[Match distance: 7  	Seq: IILLLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=IILLLLKKFLKKL)

[Match distance: 7  	Seq: IWLLLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=IWLLLLKKFLKKL)

[Match distance: 7  	Seq: LKKLKKLLKKLKKKL](https://www.dbaasp.org/search?sequence.value=LKKLKKLLKKLKKKL)

[Match distance: 7  	Seq: LKLKLKLKLKLKLKL](https://www.dbaasp.org/search?sequence.value=LKLKLKLKLKLKLKL)

## Peptide : peptide_803 	Seq: WLILPKLKCLLKKL
[Match distance: 4  	Seq: KLLLPKLKGLLFKL](https://www.dbaasp.org/search?sequence.value=KLLLPKLKGLLFKL)

[Match distance: 5  	Seq: LLKKPLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=LLKKPLLKKLLKKL)

[Match distance: 6  	Seq: ALLSLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=ALLSLLKKLLKKL)

[Match distance: 6  	Seq: GLLKKLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=GLLKKLLKKLLKKL)

[Match distance: 6  	Seq: GLLSLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=GLLSLLKKLLKKL)

[Match distance: 6  	Seq: KWKKPLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=KWKKPLLKKLLKKL)

[Match distance: 6  	Seq: LLPKLKGLLFKL](https://www.dbaasp.org/search?sequence.value=LLPKLKGLLFKL)

[Match distance: 6  	Seq: WILLLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=WILLLLKKFLKKL)

[Match distance: 6  	Seq: WKKLKKLLKKLKKL](https://www.dbaasp.org/search?sequence.value=WKKLKKLLKKLKKL)

[Match distance: 7  	Seq: ALPSLLKKLLKKL](https://www.dbaasp.org/search?sequence.value=ALPSLLKKLLKKL)

## Peptide : peptide_1197 	Seq: KVLLLLCKLKKK
[Match distance: 5  	Seq: KKKLLLLLKKK](https://www.dbaasp.org/search?sequence.value=KKKLLLLLKKK)

[Match distance: 5  	Seq: KLKKLLKKLLKK](https://www.dbaasp.org/search?sequence.value=KLKKLLKKLLKK)

[Match distance: 5  	Seq: KLLLLLLCKKKKC](https://www.dbaasp.org/search?sequence.value=KLLLLLLCKKKKC)

[Match distance: 5  	Seq: KWKLLLLLLKWK](https://www.dbaasp.org/search?sequence.value=KWKLLLLLLKWK)

[Match distance: 6  	Seq: IILLLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=IILLLLKKFLKKL)

[Match distance: 6  	Seq: IILLLLKKFLKKW](https://www.dbaasp.org/search?sequence.value=IILLLLKKFLKKW)

[Match distance: 6  	Seq: IWLLLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=IWLLLLKKFLKKL)

[Match distance: 6  	Seq: KFFRKLKKLVKK](https://www.dbaasp.org/search?sequence.value=KFFRKLKKLVKK)

[Match distance: 6  	Seq: KILKLLKKFLKKL](https://www.dbaasp.org/search?sequence.value=KILKLLKKFLKKL)

[Match distance: 6  	Seq: KKLLLGLGLLLKK](https://www.dbaasp.org/search?sequence.value=KKLLLGLGLLLKK)

## Peptide : peptide_946 	Seq: LVRIKIRPIIRPIIR
[Match distance: 8  	Seq: IIRRIIRRIIRRIIRR](https://www.dbaasp.org/search?sequence.value=IIRRIIRRIIRRIIRR)

[Match distance: 8  	Seq: ILPIKIPIIPIRR](https://www.dbaasp.org/search?sequence.value=ILPIKIPIIPIRR)

[Match distance: 8  	Seq: LRRIIRKIIHIIK](https://www.dbaasp.org/search?sequence.value=LRRIIRKIIHIIK)

[Match distance: 8  	Seq: LRRIIRKIIHIIKK](https://www.dbaasp.org/search?sequence.value=LRRIIRKIIHIIKK)

[Match distance: 9  	Seq: IRIRIRPIRIRIRP](https://www.dbaasp.org/search?sequence.value=IRIRIRPIRIRIRP)

[Match distance: 9  	Seq: IRPIIRPIIRPIIRPI](https://www.dbaasp.org/search?sequence.value=IRPIIRPIIRPIIRPI)

[Match distance: 9  	Seq: IRRIIRKIIHIIKK](https://www.dbaasp.org/search?sequence.value=IRRIIRKIIHIIKK)

[Match distance: 9  	Seq: KIIIKIKKKIKIIIK](https://www.dbaasp.org/search?sequence.value=KIIIKIKKKIKIIIK)

[Match distance: 9  	Seq: KIIKKIIKIIKKIIK](https://www.dbaasp.org/search?sequence.value=KIIKKIIKIIKKIIK)

[Match distance: 10  	Seq: FFRKVLKLIRKIWR](https://www.dbaasp.org/search?sequence.value=FFRKVLKLIRKIWR)

