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

# Setting up data
function load_and_crop_image(filename; target_size=(48, 48))
    img = load(filename)
    h_start = (size(img, 1) - target_size[1]) ÷ 2 + 1
    w_start = (size(img, 2) - target_size[2]) ÷ 2 + 1
    return img[h_start:(h_start + target_size[1] - 1), w_start:(w_start + target_size[2] - 1)]
end

function extract_alpha(fname::String)
    m = match(r"_a([0-9]+(?:\.[0-9]+)?)\.png$", fname)
    return m === nothing ? 0.0f0 : parse(Float32, m.captures[1])
end

img_dir = raw"/home/ck422/UROP_2025/Astowell.jl-main/Astowell.jl-main/notebooks/Fractal_img_50_ext/"
#img_dir = raw"/rds/general/user/ck422/home/Astowell.jl-main/Astowell.jl-main/notebooks/Fractal_img_50_ext/"
img_files = filter(f -> endswith(f, ".png"), readdir(img_dir))
n = length(img_files)
alpha_values = [extract_alpha(fname) for fname in img_files]

sample_img = load_and_crop_image(joinpath(img_dir, img_files[1]))
H, W = size(sample_img)
C = size(channelview(sample_img), 1)
n_image = H * W * C

X_for_norm = Array{Float32}(undef, n_image, n)
for (i, fname) in enumerate(img_files)
    img_obj = load_and_crop_image(joinpath(img_dir, fname))
    img_array_hwc = permutedims(channelview(img_obj), (2, 3, 1))
    X_for_norm[:, i] = vec(Float32.(img_array_hwc))
end
data_mean = mean(X_for_norm)
data_std = std(X_for_norm)

# ========== Model Definition ==========
nvars = n_image
nconds = 1
naugs = 0
n_in = nvars + naugs
hidden_dim = 64
start_H, start_W = H ÷ 4, W ÷ 4

function ResBlock(channels)
    return SkipConnection(
        Chain(
            GroupNorm(channels, channels ÷ 4),
            leakyrelu,
            Conv((3, 3), channels => channels; pad=1),
            GroupNorm(channels, channels ÷ 4),
            leakyrelu
        ),
        +
    )
end

nn = Chain(
    Dense(n_in + nconds => start_H * start_W * hidden_dim),
    Lux.WrappedFunction(x -> reshape(x, start_H, start_W, hidden_dim, size(x, 2))),
    ResBlock(hidden_dim),
    ResBlock(hidden_dim),
    ConvTranspose((4, 4), hidden_dim => hidden_dim ÷ 2; stride=2, pad=1),
    GroupNorm(hidden_dim ÷ 2, (hidden_dim ÷ 2) ÷ 4),
    leakyrelu,
    ConvTranspose((4, 4), hidden_dim ÷ 2 => C; stride=2, pad=1),
    Lux.WrappedFunction(MLUtils.flatten)
)

icnf = construct(
    FFJORD,
    nn,
    nvars,
    naugs;
    compute_mode = LuxVecJacMatrixMode(AutoZygote()),
    device = gdev,
    tspan = (0.0f0, 0.5f0),
    sol_kwargs = (;
        save_everystep = false,
        alg = BS3(),
        sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
        abstol = 1f-4,
        reltol = 1f-4,
        maxiters = 500
    )
)
ps, st = Lux.setup(Random.default_rng(), icnf)
ps = gdev(ComponentArray(ps))
st = gdev(st)

# ========== Loss Function ==========

function ffjord_loss(xs, ys, ps_current, zrs, ϵ)
    nn_cond = ContinuousNormalizingFlows.CondLayer(icnf.nn, ys)
    prob = SciMLBase.ODEProblem(
        SciMLBase.ODEFunction((u, p, t) -> ContinuousNormalizingFlows.augmented_f(u, p, t, icnf, TrainMode(), nn_cond, st, ϵ)),
        vcat(xs, zrs), icnf.tspan, ps_current
    )
    sol = solve(prob; icnf.sol_kwargs...).u[end]
    z = sol[begin:(end - (3 + naugs)), :]
    Δlogp = sol[end - (2 + naugs), :]
    logpz = cu(logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp .- (nvars * log(data_std))
    return -mean(logp̂x)
end

# ========== Training ==========

batchsize = 64
initial_lr = 1e-4
scheduler = Exp(initial_lr, 0.999)
min_lr = 1e-6
opt = Optimisers.AdamW(initial_lr)
opt_state = Optimisers.setup(opt, ps)

checkpoint_dir = "model_cnn_checkpoint"
mkpath(checkpoint_dir)
n_epochs = 5000

# Preallocate GPU buffers
zrs_buffer = CUDA.zeros(Float32, naugs + 3, batchsize)
ϵ_buffer = CUDA.zeros(Float32, nvars, batchsize)

@info "Training started"

for epoch in 1:n_epochs
    global ps, opt_state

    new_lr = max(scheduler(epoch), min_lr)
    Optimisers.adjust!(opt_state, new_lr)

    shuffled_indices = randperm(n)
    total_loss = 0.0
    num_batches = ceil(Int, n / batchsize)

    for i in 1:num_batches
        batch_indices = shuffled_indices[((i-1)*batchsize + 1):min(i*batchsize, n)]
        current_batch_size = length(batch_indices)

        x_batch_cpu = Array{Float32}(undef, n_image, current_batch_size)

        for (j, idx) in enumerate(batch_indices)
            img = load_and_crop_image(joinpath(img_dir, img_files[idx]))

            # Manual augmentation
            if rand() < 0.5
                img = reverse(img; dims=2)
            end
            angle = deg2rad(rand(-10:10))
            img_rotated = ImageTransformations.imrotate(img, angle, fill=0)

            h_rot, w_rot = size(img_rotated)
            h_start = (h_rot - H) ÷ 2 + 1
            w_start = (w_rot - W) ÷ 2 + 1
            img_final = img_rotated[h_start:(h_start + H - 1), w_start:(w_start + W - 1)]

            img_array_hwc = permutedims(channelview(img_final), (2, 3, 1))
            x_batch_cpu[:, j] = vec(Float32.(img_array_hwc))
        end

        @inbounds @. x_batch_cpu = (x_batch_cpu - data_mean) / data_std
        y_batch_cpu = reshape(alpha_values[batch_indices], 1, :)

        x_gpu = gdev(x_batch_cpu)
        y_gpu = gdev(y_batch_cpu)

        zrs = zrs_buffer[:, 1:current_batch_size]
        ϵ = ϵ_buffer[:, 1:current_batch_size]
        ϵ .= CUDA.randn(Float32, nvars, current_batch_size)

        loss, grads = Zygote.withgradient(p -> ffjord_loss(x_gpu, y_gpu, p, zrs, ϵ), ps)
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
        @info "Saved/$save_path"
    end
end

@info "done"
