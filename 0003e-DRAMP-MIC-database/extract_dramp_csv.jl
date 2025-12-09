using CSV
using DataFrames
using Unitful

# Amino acid residue masses (average, in Da) - mass of AA minus H2O lost in peptide bond
const AA_RESIDUE_MW = Dict{Char, Float64}(
    'A' => 71.08,  'R' => 156.19, 'N' => 114.10, 'D' => 115.09, 'C' => 103.14,
    'E' => 129.12, 'Q' => 128.13, 'G' => 57.05,  'H' => 137.14, 'I' => 113.16,
    'L' => 113.16, 'K' => 128.17, 'M' => 131.20, 'F' => 147.18, 'P' => 97.12,
    'S' => 87.08,  'T' => 101.10, 'W' => 186.21, 'Y' => 163.18, 'V' => 99.13,
    'U' => 150.04, 'O' => 237.30  # Selenocysteine, Pyrrolysine
)

function peptide_mw(sequence::AbstractString)::Float64
    seq = uppercase(sequence)
    mw = 18.015  # H2O for free N/C termini
    for aa in seq
        mw += get(AA_RESIDUE_MW, aa, 110.0)  # 110 Da fallback for unknown AAs
    end
    return mw
end

# Function to parse MIC values with units from DRAMP format
function parse_mic(mic_string::AbstractString)
    # Handle missing or empty values
    if ismissing(mic_string) || isempty(strip(mic_string))
        return missing
    end
    
    mic_string = strip(mic_string)
    
    # DRAMP MIC format: "Species name (MIC=value unit)"
    # Extract number and unit using regex
    # Matches patterns like: MIC=0.304 μg/ml, MIC≤0.13 μg/ml, MIC=2 mg/mL, MIC=5 mM
    match_result = match(r"MIC[=≤≥<>~]*\s*([\d.]+)\s*([μum]?[Gg]/[mM][lL]|[μum]?[MmGg])", mic_string)
    
    if match_result === nothing
        return missing
    end
    
    value = tryparse(Float64, match_result.captures[1])
    if value === nothing
        @warn "Could not parse MIC value: $(match_result.captures[1]) in $mic_string"
        return missing
    end
    unit_str = match_result.captures[2]
    
    # Normalize unit string and convert to Unitful
    unit_str_lower = lowercase(unit_str)
    
    if occursin("μg/ml", unit_str_lower) || occursin("ug/ml", unit_str_lower)
        return value * u"μg/mL"
    elseif occursin("mg/ml", unit_str_lower)
        return value * u"mg/mL"
    elseif occursin("μm", unit_str_lower) || occursin("um", unit_str_lower)
        return value * u"μM"
    elseif occursin("mm", unit_str_lower)
        return value * u"mM"
    elseif occursin("μg", unit_str_lower) || occursin("ug", unit_str_lower)
        return value * u"μg/mL"  # Assume per mL if not specified
    elseif occursin("mg", unit_str_lower)
        return value * u"mg/mL"
    else
        @warn "Unknown unit: $unit_str in $mic_string"
        return missing
    end
end

# Function to extract individual organism-MIC pairs from Target_Organism field
function extract_mic_entries(target_organism_string::AbstractString)
    # Split by comma, but be careful with commas inside parentheses
    entries = Tuple{String, Any}[]
    
    # Find all organism entries with MIC values
    # Pattern: "Organism name (MIC=value unit)"
    pattern = r"([^,(]+?)\s*\(MIC[=≤≥<>~]*\s*[\d.]+\s*[μum]?[MmGg](?:/[mM][lL])?\)"
    
    for m in eachmatch(pattern, target_organism_string)
        full_match = m.match
        organism = strip(m.captures[1])
        mic_value = parse_mic(full_match)
        push!(entries, (organism, mic_value))
    end
    
    return entries
end

# Convert single MIC value to μg/mL using peptide MW for molar units
function mic_to_ug_per_ml(mic, mw::Float64)::Union{Missing, Float64}
    ismissing(mic) && return missing
    mic_unit = unit(mic)
    mic_val = ustrip(mic)
    
    if mic_unit == u"μg/mL"
        return mic_val
    elseif mic_unit == u"mg/mL"
        return mic_val * 1000.0
    elseif mic_unit == u"μM"
        return mic_val * mw / 1000.0
    elseif mic_unit == u"mM"
        return mic_val * mw
    else
        return missing
    end
end

# Geometric mean, ignoring missing values
geomean(x) = exp(sum(log, x) / length(x))

# Function to extract DRAMP data from CSV
function extract_dramp_data(csv_file::String)
    df = CSV.read(csv_file, DataFrame, stringtype=String)
    
    println("Available columns: ", names(df))
    println("Total entries in CSV: ", nrow(df))
    
    result = DataFrame(
        DRAMP_ID = String[],
        Sequence = String[],
        Sequence_Length = Int[],
 #       Name = String[],
        n_organisms = Int[],
        MIC_geomean_ug_mL = Union{Missing, Float64}[]
    )
    
    for row in eachrow(df)
        dramp_id = row.DRAMP_ID
        sequence = row.Sequence
        seq_length = row.Sequence_Length
        name = row.Name
        target_org_field = row.Target_Organism
        
        mic_entries = extract_mic_entries(target_org_field)
        
        if !isempty(mic_entries)
            mw = peptide_mw(sequence)
            # Convert all MIC values to μg/mL and filter out missing
            mic_vals = Float64[]
            for (_, mic_val) in mic_entries
                converted = mic_to_ug_per_ml(mic_val, mw)
                !ismissing(converted) && converted > 0 && push!(mic_vals, converted)
            end
            
            if !isempty(mic_vals)
                push!(result, (
                    string(dramp_id),
                    string(sequence),
                    Int(seq_length),
#                    string(name),
                    length(mic_vals),
                    geomean(mic_vals)
                ))
            end
        end
    end
    
    return result
end

function main()
    csv_file = "Antibacterial_amps.csv"
    
    println("=" ^ 60)
    println("DRAMP Data Extraction Tool")
    println("WARNING! Its a bit more complex than this - some of the AMPs have long lipid chains etc.; and I'm just geometric meaning the total activity converted to micrograms/ml, and god knows whether that is in anyway reliable...HERE BE DRAGONS!")
    println("=" ^ 60)
    
    data = extract_dramp_data(csv_file)
    
    println("\n✓ Extracted $(nrow(data)) peptides with MIC data")
    println("\nFirst 10 entries:")
    println(first(data, 10))
    
    # Save processed data
    output_file = "dramp_geometric_MIC.csv"
    CSV.write(output_file, data)
    println("\n✓ Processed data saved to $output_file")
    
    # Summary statistics
    println("\n" * "=" ^ 60)
    println("Summary Statistics")
    println("=" ^ 60)
    println("Total peptides: ", nrow(data))
    println("Total organism-MIC measurements: ", sum(data.n_organisms))
    println("MIC range (μg/mL): ", round(minimum(data.MIC_geomean_ug_mL), digits=3), " - ", round(maximum(data.MIC_geomean_ug_mL), digits=3))
    
    return data
end

main()

