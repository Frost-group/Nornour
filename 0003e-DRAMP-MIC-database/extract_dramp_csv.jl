using CSV
using DataFrames

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

# Extract organism-MIC pairs from Target_Organism field and convert to μg/mL
function extract_mic_entries(target_organism::AbstractString, mw::Float64; verbose::Bool=false)
    entries = Tuple{String, Union{Missing, Float64}}[]
    pattern = r"([^,(]+?)\s*\(MIC[=≤≥<>~]*\s*([\d.]+)\s*([^)]+)\)"
    
    for m in eachmatch(pattern, target_organism)
        organism = strip(m.captures[1])
        value = tryparse(Float64, m.captures[2])
        unit_raw = strip(m.captures[3])
        unit_str = lowercase(replace(replace(unit_raw, 'µ' => 'μ'), ' ' => ""))  # normalize
        
        verbose && println("Organism: $organism, Value: $value, Unit: $unit_raw")
        
        if value === nothing
            push!(entries, (organism, missing))
            continue
        end
        
        mic_ug_ml = if occursin(r"[μu]g/m", unit_str) || occursin("mg/l", unit_str)
            value
        elseif occursin("ng/ml", unit_str)
            value / 1000.0
        elseif occursin("mg/ml", unit_str)
            value * 1000.0
        elseif occursin("pm", unit_str) || occursin("pmol", unit_str)
            value * mw / 1e9
        elseif occursin("nm", unit_str)
            value * mw / 1e6
        elseif occursin(r"[μu]m", unit_str) || occursin("microm", unit_str)
            value * mw / 1000.0
        elseif occursin("mm", unit_str)
            value * mw
        else
            @warn "Unhandled unit '$unit_raw' for =$value $organism - skipping"
            missing
        end

        verbose && println("Attempted unit conversion Value: $value, Unit: $unit_raw, Converted to: $mic_ug_ml")
        
        push!(entries, (organism, mic_ug_ml))
    end
    
    return entries
end

# Geometric mean, ignoring missing values
geomean(x) = exp(sum(log, x) / length(x))

# Function to extract DRAMP data from CSV
function extract_dramp_data(csv_file::String; verbose::Bool=false)
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
    
    for (i, row) in enumerate(eachrow(df))
        dramp_id = row.DRAMP_ID
        sequence = row.Sequence
        seq_length = row.Sequence_Length
        name = row.Name
        target_org_field = row.Target_Organism
       
        if verbose
            println("DRAMP_ID: $dramp_id")
            println("Sequence: $sequence")
            println("Sequence_Length: $seq_length")
            println("Name: $name")
            println("Target_Organism (should contain MICs): $target_org_field")
        end

        ismissing(sequence) && continue
        mw = peptide_mw(sequence)
        mic_entries = extract_mic_entries(target_org_field, mw, verbose=verbose)
        
        if !isempty(mic_entries)
            mic_vals = [v for (_, v) in mic_entries if !ismissing(v) && v > 0]
        
            if verbose && !isempty(mic_vals)
                println("OK, extracted $(length(mic_vals)) MICs as $mic_vals µg/ml,  geomean $(geomean(mic_vals)) µg/ml")
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

        verbose && i > 200 && break # just do first N for println debugging of unit conversions 
    end
    
    return result
end

function main()
    csv_file = "Antibacterial_amps.csv"
    
    println("=" ^ 60)
    println("DRAMP Data Extraction Tool")
    println("WARNING! Its a bit more complex than this - some of the AMPs have long lipid chains etc.; and I'm just geometric meaning the total activity converted to micrograms/ml, and god knows whether that is in anyway reliable...HERE BE DRAGONS!")
    println("=" ^ 60)
    
    data = extract_dramp_data(csv_file, verbose=false)
    
    println("\n✓ Extracted $(nrow(data)) peptides with MIC data")
    println("\nFirst 10 entries:")
    println(first(data, 10))

    println("\nTop 10 entries by MIC value:")
    println(first(sort(data, :MIC_geomean_ug_mL), 10))
    
    # Save processed data
    output_file = "dramp_geometric_MIC.csv"
    CSV.write(output_file, data)
    println("\n✓ Processed data saved to $output_file")
    
    # Summary statistics
    println("\n" * "=" ^ 60)
    println("Summary Statistics")
    println("=" ^ 60)
    println("Total peptides with some MIC data: ", nrow(data))
    println("Total organism-MIC measurements: ", sum(data.n_organisms))
    println("MIC range (μg/mL): ", round(minimum(data.MIC_geomean_ug_mL), digits=3), " - ", round(maximum(data.MIC_geomean_ug_mL), digits=3))
    
    return data
end

main()

