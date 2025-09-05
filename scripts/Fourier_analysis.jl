# Code Quantitatively compares my generated images to the training images
# Calculates power spectra for both sets of images 
# Then radially averages the power spectra into 1d plots and prints a correlation_score

using FFTW, Plots, Images, FileIO, Statistics, Printf

# --- Original Function ---
function get_power_spectrum(img_path::String)
    """
    Loads an image, converts it to grayscale, and computes its 2D power spectrum.
    """
    # Loads and converts images to grey scale
    img_color = load(img_path)
    img_gray = Gray.(img_color)
    img_array = float.(channelview(img_gray))
    f_transform = fft(img_array)

    f_transform_shifted = fftshift(f_transform) # shifts zero frequency component
    # Calculates power spectrum
    power_spectrum = abs.(f_transform_shifted) .^ 2
    
    return power_spectrum
end

#  Computes the radially averaged power spectrum from a 2D power spectrum.
function radially_average(spectrum::Matrix{<:AbstractFloat})
    ny, nx = size(spectrum)
    center_x, center_y = (nx ÷ 2) + 1, (ny ÷ 2) + 1

    # Create a grid of distances from the center
    x = (1:nx) .- center_x
    y = (1:ny) .- center_y
    distances = sqrt.(x' .^ 2 .+ y .^ 2)

    # Bin distances into integer radi
    max_radius = Int(floor(min(center_x, center_y) / sqrt(2))) # Ensure we stay within the image bounds
    radii_bins = 1:max_radius
    
    power_1d = zeros(length(radii_bins))

    for r in radii_bins
        # Find all pixels within this radial bin
        mask = (distances .>= r - 1) .& (distances .< r)
        if any(mask)
            power_1d[r] = mean(spectrum[mask])
        end
    end
    
    return radii_bins, power_1d
end


# define paths
path_target = raw"C:\Users\casan\Downloads\UROP_2025\Astowell.jl-main\Astowell.jl-main\notebooks\Zannen_train\Comparable_img\50_img_a0.5009.png"
path_generated = raw"C:\Users\casan\Downloads\UROP_2025\Machine_learned_test_img\alpha_0.5.png"

power_spec_target = get_power_spectrum(path_target)
power_spec_generated = get_power_spectrum(path_generated)

# Visualise 2D results on a log scale plot
p1 = heatmap(
    log10.(power_spec_target),
    aspect_ratio = 1,
    c = :viridis,
    title = "Target Power Spectrum (log scale) α = 0.5",
    colorbar_title = "log10(Power)",
    xaxis = false, yaxis = false, xticks = false, yticks = false
)

p2 = heatmap(
    log10.(power_spec_generated),
    aspect_ratio = 1,
    c = :viridis,
    title = "Generated Power Spectrum (log scale) α = 0.5",
    colorbar_title = "log10(Power)",
    xaxis = false, yaxis = false, xticks = false, yticks = false
)

plot2d = plot(p1, p2, layout = (1, 2), size = (1200, 500))
display(plot2d) 
savefig(plot2d, "power_spectra_comparison_alpha_0.5.png")

# Perform Radial Averaging
k_target, Pk_target = radially_average(power_spec_target)
k_generated, Pk_generated = radially_average(power_spec_generated)

# 2. Calculate the Correlation Coefficient
min_len = min(length(Pk_target), length(Pk_generated))
correlation_score = cor(Pk_target[1:min_len], Pk_generated[1:min_len])

# looks pretty :)
println("------------------------------------------")
println("Correlation Score: ", correlation_score)
println("------------------------------------------")

# Plots the 1D spectra 
plot1d = plot(k_target, Pk_target,
    label="Target",
    xlabel="Spatial Frequency (k)",
    ylabel="Power",
    title="Radially Averaged Power Spectra (α = 0.5)",
    xscale=:log10, yscale=:log10,
    legend=:bottomleft,
    linewidth=2)

plot!(plot1d, k_generated, Pk_generated, 
      label="Generated with correlation: $(@sprintf("%.7g", correlation_score))", 
      linestyle=:dash, 
      linewidth=2)
display(plot1d)
savefig(plot1d, "radially_averaged_comparison_alpha_0.5.png")
