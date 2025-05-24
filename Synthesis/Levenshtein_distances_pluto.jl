### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 971a6d0c-8f82-4602-bf12-f2c60c815def
md"""
# Levenshtein

Adapted from https://github.com/rawrgrr/Levenshtein.jl
"""

# ╔═╡ 6193504c-d747-4661-ab4d-49a5b736fa60
function levenshtein(source::AbstractString, target::AbstractString)
    return levenshtein(source, target, 1)
end

# ╔═╡ cc519ed3-a546-4f9e-8938-22671b050497
function levenshtein(source::AbstractString, target::AbstractString, cost::Real)
    return levenshtein(source, target, cost, cost, cost)
end

# ╔═╡ fc98c9fd-f819-4b59-97b1-39b3ee6d5709
function levenshtein!(
    source::AbstractString,
    target::AbstractString,
    deletion_cost::R,
    insertion_cost::S,
    substitution_cost::T,
    costs::Matrix = Array{promote_type(R, S, T)}(undef, 2, length(target) + 1)
) where {R<:Real,S<:Real,T<:Real}
    cost_type = promote_type(R,S,T)
    if length(source) < length(target)
        # Space complexity of function = O(length(target))
        return levenshtein!(target, source, insertion_cost, deletion_cost, substitution_cost, costs)
    else
        if length(target) == 0
            return length(source) * deletion_cost
        else
            old_cost_index = 1
            new_cost_index = 2

            costs[old_cost_index, 1] = 0
            for i in 1:length(target)
                costs[old_cost_index, i + 1] = i * insertion_cost
            end

            i = 0
            for r in source
                i += 1

                # Delete i characters from source to get empty target
                costs[new_cost_index, 1] = i * deletion_cost

                j = 0
                for c in target
                    j += 1

                    deletion = costs[old_cost_index, j + 1] + deletion_cost
                    insertion = costs[new_cost_index, j] + insertion_cost
                    substitution = costs[old_cost_index, j]
                    if r != c
                        substitution += substitution_cost
                    end

                    costs[new_cost_index, j + 1] = min(deletion, insertion, substitution)
                end

                old_cost_index, new_cost_index = new_cost_index, old_cost_index
            end

            new_cost_index = old_cost_index
            return costs[new_cost_index, length(target) + 1]
        end
    end
end

# ╔═╡ 9481a8f3-a189-4b0e-b699-15dcd37ab86c
function levenshtein(source::AbstractString, target::AbstractString, deletion_cost::R, insertion_cost::S, substitution_cost::T) where {R<:Real,S<:Real,T<:Real}
    return levenshtein!(target, source, insertion_cost, deletion_cost, substitution_cost)
end

# ╔═╡ dbbb094e-03d0-4fef-9548-666257b21fe4
db=readlines("/Users/jmf02/REPOS/Nornour/0003d-DBAASP-Database/Database_of_Antimicrobial_Activity_and_structure_of_Peptides")

# ╔═╡ dd1e1ca1-d018-4d6f-bef6-b826cf8c9289
#peptide_10 IKFLSLKLGLSLKK
#peptide_11 LILKPLKLLKCLKKL
target="LILKPLKLLKCLKKL"

# ╔═╡ 5984685f-b993-4dd2-830e-9be2beb4b8ed
ls=[levenshtein(target,pep, 5,5,1) for pep in db]

# ╔═╡ e1f6b696-74ee-4247-aeea-d43313511fe2
m,idx=findmin(ls)

# ╔═╡ 8a874edc-44df-408c-89c7-e6486fd126a8
db[idx]

# ╔═╡ 9096b414-aaff-4384-b5eb-72d420321cfa
sortslices(hcat(ls,db), dims=1)

# ╔═╡ 0fc59863-f695-45fd-8347-c69863e06cfe
### Hold my beer

# ╔═╡ b2dbaf84-2ef4-4811-8df1-47d5c744e96a
parse_peptides(file) = [(m[1], m[2]) for m in eachmatch(r">(\w+)\n([A-Z]+)", read(file, String))]

# ╔═╡ 3121e65a-13b6-42c1-8c31-12dcb98ba5e0
synth="""
#First round of synthesis - peptides picked from the top 20 LSTM predicted peps sorted by AMP probability

>peptide_1
IVFLKLPAGKKKIL
Ile Val Phe Leu Lys Leu Pro Ala Gly Lys Lys Lys Ile Leu
14 amino acids; Mw=1568.17

>peptide_8
IVLPKLKCLLIK
Ile Val Leu Pro Lys Leu Lys Cys Leu Leu Ile Lys
12 amino acids; Mw=1380.98

>peptide_10
IKFLSLKLGLSLKK
Ile Lys Phe Leu Ser Leu Lys Leu Gly Leu Ser Leu Lys Lys
14 amino acids; Mw=1588.16

>peptide_11
LILKPLKLLKCLKKL
Leu Ile Leu Lys Pro Leu Lys Leu Leu Lys Cys Leu Lys Lys Leu
15 amino acids; Mw=1764.54

#Second Round of pep synth - first 6 taken from top 20 peps sorted by AMP probability (all > 0.995), second six chosen from list sorted by lowest MIC values - named by ID tag with associated AMP probability listed.


>peptide_8
IVLPKLKCLLIK
Ile Val Leu Pro Lys Leu Lys Cys Leu Leu Ile Lys
12 amino acids; Mw=1380.98

>peptide_10
IKFLSLKLGLSLKK
Ile Lys Phe Leu Ser Leu Lys Leu Gly Leu Ser Leu Lys Lys
14 amino acids; Mw=1588.16

>peptide_11
LILKPLKLLKCLKKL
Leu Ile Leu Lys Pro Leu Lys Leu Leu Lys Cys Leu Lys Lys Leu
15 amino acids; Mw=1764.54

>peptide_13
ILFIGSLIKKRPIK
Ile Leu Phe Ile Gly Ser Leu Ile Lys Lys Arg Pro Ile Lys
14 amino acids; Mw=1626.22

>peptide_16
LWFIVKKLASKVLP
Leu Trp Phe Ile Val Lys Lys Leu Ala Ser Lys Val Leu Pro
14 amino acids; Mw=1642.19

>peptide_17
FRFPRIGIIILAVKK
Phe Arg Phe Pro Arg Ile Gly Ile Ile Ile Leu Ala Val Lys Lys
15 amino acids; Mw=1771.39




>peptide_97
VFFWLLCKLKKKLL
Val Phe Phe Trp Leu Leu Cys Lys Leu Lys Lys Lys Leu Leu
0.9395, 14 amino acids; Mw= 1779.44Da

>peptide_1181
VLKCLCLKLKKKLL
Val Leu Lys Cys Leu Cys Leu Lys Leu Lys Lys Lys Leu Leu
0.987, 14 amino acids; Mw= 1643.36

>peptide_1304
WLLLLCLKKCLKKKL
Trp Leu Leu Leu Leu Cys Leu Lys Lys Cys Leu Lys Lys Lys Leu
0.982, 15 amino acids; Mw= 1843.6

>peptide_803
WLILPKLKCLLKKL
Trp Leu Ile Leu Pro Lys Leu Lys Cys Leu Leu Lys Lys Leu
0.9935, 14 amino acids; Mw= 1709.4

>peptide_1197
KVLLLLCKLKKK
Lys Val Leu Leu Leu Leu Cys Lys Leu Lys Lys Lys
0.9385, 12 amino acids; Mw= 1427.05

>peptide_946
LVRIKIRPIIRPIIR
Leu Val Arg Ile Lys Ile Arg Pro Ile Ile Arg Pro Ile Ile Arg
0.9175, 15 amino acids; Mw= 1856.57
"""

# ╔═╡ dcbed47d-bc84-491c-aec0-8447788e08b9
peptides=[(m[1], m[2]) for m in eachmatch(r">(\w+)\n([A-Z]+)", synth)]

# ╔═╡ cfd1b43a-2cba-4740-9a4f-267834faefa6
for pep in peptides
	println("Peptide : $(pep[1]) \tSeq: $(pep[2])")

	ls=[levenshtein(pep[2],p) for p in db]
	scores=unique(sortslices(hcat(ls,db), dims=1),dims=1)

	for i in 1:10
		println("    Match distance: $(scores[i,:][1])  \tSeq: $(scores[i,:][2])")
	end
end

# ╔═╡ 7751afa6-170a-4979-b3b4-d62ed8adfe55


# ╔═╡ fd8b68c3-38f8-4d1b-8ad1-36c6b5bdb287
for pep in peptides
	println("## Peptide : $(pep[1]) \tSeq: $(pep[2])")

# putting a cost of 2x for addition and deletion of codons vs. substitution:
	ls=[levenshtein(pep[2],p, 2,2,1) for p in db]
	scores=unique(sortslices(hcat(ls,db), dims=1),dims=1)

	# print-out as Markdown for copy+pasting into Zulip, with link to DBAASP
	for i in 1:10
		URL="https://www.dbaasp.org/search?sequence.value=$(scores[i,:][2])"
		
		println("[Match distance: $(scores[i,:][1])  \tSeq: $(scores[i,:][2])]($(URL))")
	end
end

# ╔═╡ f5cfaf6a-2d8a-46ee-8cde-bcd349b80147


# ╔═╡ 1505c061-9328-473c-8f94-187d6193184f


# ╔═╡ 056a915f-83f1-476e-9538-ff27b5f80078


# ╔═╡ 81efd062-d896-490e-bd1e-24b94234f9b0


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─971a6d0c-8f82-4602-bf12-f2c60c815def
# ╠═6193504c-d747-4661-ab4d-49a5b736fa60
# ╠═cc519ed3-a546-4f9e-8938-22671b050497
# ╠═9481a8f3-a189-4b0e-b699-15dcd37ab86c
# ╠═fc98c9fd-f819-4b59-97b1-39b3ee6d5709
# ╠═dbbb094e-03d0-4fef-9548-666257b21fe4
# ╠═dd1e1ca1-d018-4d6f-bef6-b826cf8c9289
# ╠═5984685f-b993-4dd2-830e-9be2beb4b8ed
# ╠═e1f6b696-74ee-4247-aeea-d43313511fe2
# ╠═8a874edc-44df-408c-89c7-e6486fd126a8
# ╠═9096b414-aaff-4384-b5eb-72d420321cfa
# ╠═0fc59863-f695-45fd-8347-c69863e06cfe
# ╠═b2dbaf84-2ef4-4811-8df1-47d5c744e96a
# ╟─3121e65a-13b6-42c1-8c31-12dcb98ba5e0
# ╠═dcbed47d-bc84-491c-aec0-8447788e08b9
# ╠═cfd1b43a-2cba-4740-9a4f-267834faefa6
# ╠═7751afa6-170a-4979-b3b4-d62ed8adfe55
# ╠═fd8b68c3-38f8-4d1b-8ad1-36c6b5bdb287
# ╠═f5cfaf6a-2d8a-46ee-8cde-bcd349b80147
# ╠═1505c061-9328-473c-8f94-187d6193184f
# ╠═056a915f-83f1-476e-9538-ff27b5f80078
# ╠═81efd062-d896-490e-bd1e-24b94234f9b0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
