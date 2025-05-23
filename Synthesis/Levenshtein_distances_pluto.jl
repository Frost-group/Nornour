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
ls=[levenshtein(target,pep) for pep in db]

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

# ╔═╡ 13d461c9-daad-4c54-a629-86ab59c97ffe
peptides=parse_peptides("../Synthesis/SecondRound.txt")

# ╔═╡ cfd1b43a-2cba-4740-9a4f-267834faefa6
for pep in peptides
	println("Peptide : $(pep[1]) Seq: $(pep[2])")

	ls=[levenshtein(pep[2],p) for p in db]
	scores=sortslices(hcat(ls,db), dims=1)

	for i in 1:10
		println("    Match distance: $(scores[i,:][1])  Seq: $(scores[i,:][2])")
	end
end

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
# ╠═13d461c9-daad-4c54-a629-86ab59c97ffe
# ╠═cfd1b43a-2cba-4740-9a4f-267834faefa6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
