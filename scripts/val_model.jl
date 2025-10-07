# This code is used the validate the trained models by comparing the correlation coefficients
# of the radially averaged power spectra of generated images against ground-truth images.

using FFTW, Plots, Images, FileIO, Statistics, Printf, DataFrames, CSV

# Functions unchanged from the Fourier analysis code 
function get_power_spectrum(img_path::String)
    img_color = load(img_path)
    img_gray = Gray.(img_color)
    img_array = float.(channelview(img_gray))
    f_transform = fft(img_array)
    f_transform_shifted = fftshift(f_transform)
    power_spectrum = abs.(f_transform_shifted) .^ 2
    return power_spectrum
end

function radially_average(spectrum::Matrix{<:AbstractFloat})
    ny, nx = size(spectrum)
    center_x, center_y = (nx ÷ 2) + 1, (ny ÷ 2) + 1
    distances = [sqrt((j-center_x)^2 + (i-center_y)^2) for i in 1:ny, j in 1:nx]
    max_radius = Int(floor(min(center_x, center_y) / sqrt(2)))
    radii_bins = 1:max_radius
    power_1d = zeros(length(radii_bins))

    for r in radii_bins
        mask = (distances .>= r - 1) .& (distances .< r)
        if any(mask)
            power_1d[r] = mean(spectrum[mask])
        end
    end
    return radii_bins, power_1d
end

path_ground_truth = raw"C:\Users\casan\check"

# Folders that contain models 
model_folders = [
    raw"C:\Users\casan\ingvar_test0",
    raw"C:\Users\casan\generated_img_alpha0_conditioned_cpu",
    raw"C:\Users\casan\non_cnf_test_img"
]

model_names = [
    "Conditional Model",
    "Time-Evolving Model(alpha as time)",
    "Non-CNF Model"
]

# conditional model 5 only trained up to alpha = 0.64 (Important to keep in mind)


alphas_to_test = 0.0:0.1:1.0


# Create a DataFrame to store results
results_df = DataFrame(Alpha = Float32[])
for name in model_names
    results_df[!, Symbol(name)] = Float64[]
end


# Loop over each alpha value
for alpha in alphas_to_test
    # Find the corresponding image
    # NOTE! This assumes your files have a format like '...a0.1.png'
    target_fname = "alpha_$(@sprintf("%.1f", alpha)).png" 
    path_target = joinpath(path_ground_truth, target_fname)
    # Just in case I had some problems :/
    if !isfile(path_target)
        @warn "Ground truth file not found for alpha=$alpha, skipping."
        continue
    end
    
    power_spec_target = get_power_spectrum(path_target)
    k_target, Pk_target = radially_average(power_spec_target)
    
    result_row = Dict{Symbol, Any}(:Alpha => alpha)

    # Loop over each model folder
    for (i, folder) in enumerate(model_folders)
        model_name = model_names[i]
        
        # NOTE: This assumes your generated files have a format like 'alpha_0.1.png'
        generated_fname = "alpha_$(@sprintf("%.1f", alpha)).png"
        path_generated = joinpath(folder, generated_fname)
        
        if !isfile(path_generated)
            @warn "Generated file not found for alpha=$alpha in folder $model_name, skipping."
            result_row[Symbol(model_name)] = NaN # Mark as missing
            continue
        end

        power_spec_generated = get_power_spectrum(path_generated)
        k_generated, Pk_generated = radially_average(power_spec_generated)
        
        min_len = min(length(Pk_target), length(Pk_generated))
        correlation_score = cor(Pk_target[1:min_len], Pk_generated[1:min_len])
        
        result_row[Symbol(model_name)] = correlation_score
        
        @printf("Alpha: %.1f | %-10s | Correlation: %.4f\n", alpha, model_name, correlation_score)
    end
    push!(results_df, result_row)
end

println("\nAnalysis complete.")

# --- Save and Plot Results ---


csv_path = "correlation_scores.csv"
CSV.write(csv_path, results_df)
println("Results saved to $csv_path")

plot(xlabel="Alpha (α)", ylabel="Correlation Score", title="Model Performance Comparison", legend=:bottomleft)
for name in model_names
    plot!(results_df.Alpha, results_df[!, Symbol(name)], label=name, marker=:circle, linewidth=2)
end

plot_path = "model_comparison_plot.png"
savefig(plot_path)
println("Summary plot saved to $plot_path")

display(current()) 
