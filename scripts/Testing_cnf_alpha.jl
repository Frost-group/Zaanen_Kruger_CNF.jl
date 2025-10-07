# Uses alpha as the time variable for the CNF, still starts from gaussian noise 

# THIS DOESNT WORK VERY WELL!!!!
using Logging, TerminalLoggers, Statistics
global_logger(TerminalLogger())

using ContinuousNormalizingFlows, Lux, OrdinaryDiffEq, SciMLSensitivity, ADTypes
using LuxCUDA, CUDA, cuDNN
using ComponentArrays, Distributions, Random, Zygote, Optimisers, MLUtils
using Images, FileIO, JLD2
using ParameterSchedulers
using ImageTransformations


CUDA.allowscalar(false)
gdev = gpu_device()
cpu_dev = cpu_device()

# Function to load and crop images to a consistent target size
# Cropping is necessary for the data augmentation to work properly
function load_and_crop_image(filename; target_size=(48, 48))
    img = load(filename)
    h_start = (size(img, 1) - target_size[1]) ÷ 2 + 1
    w_start = (size(img, 2) - target_size[2]) ÷ 2 + 1
    return img[h_start:(h_start + target_size[1] - 1), w_start:(w_start + target_size[2] - 1)]
end

# Function for extracting alpha values from filenames
function extract_alpha(fname::String)
    m = match(r"_a([0-9]+(?:\.[0-9]+)?)\.png$", fname)
    return m === nothing ? 0.0f0 : parse(Float32, m.captures[1])
end

# Load all image files from directory and extract alpha values
img_dir = raw"/home/ck422/UROP_2025/Astowell.jl-main/Astowell.jl-main/notebooks/Fractal_img_50_ext/"
img_files = filter(f -> endswith(f, ".png"), readdir(img_dir))
n = length(img_files)
alpha_values = [extract_alpha(fname) for fname in img_files]

# Sort images in order of alpha value 
sorted_indices = sortperm(alpha_values)
img_files = img_files[sorted_indices]
alpha_values = alpha_values[sorted_indices]

# Determine image dimensions 
sample_img = load_and_crop_image(joinpath(img_dir, img_files[1]))
H, W = size(sample_img)
C = 3  # RGB channels
n_image = H * W * C  # Total number of pixels * channels

# Normalize data for better training stability
X_for_norm = Array{Float32}(undef, n_image, n)
for (i, fname) in enumerate(img_files)
    img_obj = load_and_crop_image(joinpath(img_dir, fname))
    # Convert from HWC to flattened vector format
    img_array_hwc = permutedims(channelview(img_obj), (2, 3, 1))
    X_for_norm[:, i] = vec(Float32.(img_array_hwc))
end
# Calculate mean and std for data normalization
data_mean = mean(X_for_norm)
data_std = std(X_for_norm)

# CNF model configuration
nvars = n_image    
nconds = 0         # Not conditional on alpha as alpha is used as time
naugs = 0          
n_in = nvars + naugs
hidden_dim = 64   
start_H, start_W = H ÷ 4, W ÷ 4  

# Residual block with group normalization for better training stability
function ResBlock(channels)
    return SkipConnection(
        Chain(
            GroupNorm(channels, channels ÷ 4), leakyrelu,
            Conv((3, 3), channels => channels; pad=1),
            GroupNorm(channels, channels ÷ 4), leakyrelu
        ), +
    )
end

# Neural network architecture: Dense -> Reshape -> ResBlocks -> Upsampling
nn = Chain(
    Dense(n_in => start_H * start_W * hidden_dim),
    Lux.WrappedFunction(x -> reshape(x, start_H, start_W, hidden_dim, size(x, 2))),
    ResBlock(hidden_dim),
    ResBlock(hidden_dim),
    ConvTranspose((4, 4), hidden_dim => hidden_dim ÷ 2; stride=2, pad=1),
    GroupNorm(hidden_dim ÷ 2, (hidden_dim ÷ 2) ÷ 4),
    leakyrelu,
    ConvTranspose((4, 4), hidden_dim ÷ 2 => C; stride=2, pad=1),
    Lux.WrappedFunction(MLUtils.flatten)
)

# Construct FFJORD model
icnf = construct(
    FFJORD,
    nn,
    nvars,
    naugs;
    compute_mode = LuxVecJacMatrixMode(AutoZygote()),
    device = gdev,
    tspan = (0.0f0, 1.0f0), # Base time span which will be overridden
    sol_kwargs = (;
        save_everystep = false,
        alg = BS3(),  
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),  
        abstol = 1f-4,
        reltol = 1f-4,
    )
)

# Initialize model parameters and move them to GPU
ps, st = Lux.setup(Random.default_rng(), icnf)
ps = gdev(ComponentArray(ps))  
st = gdev(st)  

# Modified loss function with dynamic time span based on alpha values
function ffjord_loss_dynamic_tspan(xs, ps_current, st_current, zrs, ϵ, dynamic_tspan)
    prob = SciMLBase.ODEProblem(
        SciMLBase.ODEFunction((u, p, t) -> ContinuousNormalizingFlows.augmented_f(u, p, t, icnf, TrainMode(), icnf.nn, st_current, ϵ)),
        vcat(xs, zrs),  
        dynamic_tspan,  # Use alpha-dependent time span
        ps_current
    )
    # Solve the ODE to get final state
    sol = solve(prob; icnf.sol_kwargs...).u[end]
    z = sol[begin:(end - (3 + naugs)), :]  
    Δlogp = sol[end - (2 + naugs), :]      
    
    # Calculate log probability in base distribution (Gaussian)
    logpz = cu(logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp .+ (nvars * log(data_std))
    return -mean(logp̂x)  # Return negative log likelihood
end

# Training configuration with exponential learning rate decay
batchsize = 256 #
initial_lr = 1e-3
scheduler = Exp(initial_lr, 0.99)  
min_lr = 1e-6
opt = Optimisers.AdamW(initial_lr)  
opt_state = Optimisers.setup(opt, ps)


checkpoint_dir = "model_cnn_checkpoint_alpha_as_time_batched"
mkpath(checkpoint_dir)
n_epochs = 5000

#@info "Training started"

# Main training loop
for epoch in 1:n_epochs
    global ps, opt_state

    # Update learning rate according to schedule
    new_lr = max(scheduler(epoch), min_lr)
    Optimisers.adjust!(opt_state, new_lr)

    # Shuffle indices while maintaining some locality for similar alpha values
    shuffled_indices = randperm(n) 
    total_loss = 0.0
    num_batches = ceil(Int, n / batchsize)

    for i in 1:num_batches
        batch_indices = shuffled_indices[((i-1)*batchsize + 1):min(i*batchsize, n)]
        current_batch_size = length(batch_indices)

        # Load and preprocess batch of images with data augmentation
        x_batch_cpu = Array{Float32}(undef, n_image, current_batch_size)
        for (j, idx) in enumerate(batch_indices)
            img = load_and_crop_image(joinpath(img_dir, img_files[idx]))
            
            # Data augmentation for images
            if rand() < 0.5; img = reverse(img; dims=2); end
            angle = deg2rad(rand(-10:10))
            img_rotated = ImageTransformations.imrotate(img, angle, fill=0)
            
            # Crop rotated image back to original size
            h_rot, w_rot = size(img_rotated)
            h_start = max(1, (h_rot - H) ÷ 2 + 1)
            w_start = max(1, (w_rot - W) ÷ 2 + 1)
            img_final = img_rotated[h_start:min(h_start + H - 1, h_rot), w_start:min(w_start + W - 1, w_rot)]
            if size(img_final) != (H,W) img_final = imresize(img_final, (H,W)) end

            # Convert to array format and store
            img_array_hwc = permutedims(channelview(img_final), (2, 3, 1))
            x_batch_cpu[:, j] = vec(Float32.(img_array_hwc))
        end
        
        # Normalize data and move to GPU
        @inbounds @. x_batch_cpu = (x_batch_cpu - data_mean) / data_std
        x_gpu = gdev(x_batch_cpu)
        
        # Set dynamic time span based on maximum alpha in batch
        batch_alphas = alpha_values[batch_indices]
        max_alpha_in_batch = maximum(batch_alphas)

        # check for zero alpha
        if max_alpha_in_batch <= 0.0f0 continue end
        
        dynamic_tspan = (0.0f0, max_alpha_in_batch)
        
        zrs = CUDA.zeros(Float32, naugs + 3, current_batch_size)
        ϵ = CUDA.randn(Float32, nvars, current_batch_size)

        # Calculate gradients and update parameters
        loss, grads = Zygote.withgradient(p -> ffjord_loss_dynamic_tspan(x_gpu, p, st, zrs, ϵ, dynamic_tspan), ps)
        opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
        total_loss += loss
    end

    avg_loss = total_loss / num_batches
    @info "Epoch $epoch: Loss = $avg_loss (LR = $new_lr)"
    
    if epoch % 100 == 0
        ps_cpu = cpu_dev(ps)
        st_cpu = cpu_dev(st)
        save_path = joinpath(checkpoint_dir, "model_cnn_epoch_$(epoch).jld2")
        jldsave(save_path; ps=ps_cpu, st=st_cpu, H=H, W=W, C=C, data_mean=data_mean, data_std=data_std)
        @info "Saved checkpoint to $save_path"
    end
end

@info "Training complete."
