using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

using BranchChain
using ProtPlot, GLMakie
using DLProteinFormats: load, PDBSimpleFlat, sample_batched_inds, length2batch
using LearningSchedules: burnin_learning_schedule, linear_decay_schedule
using CannotWaitForTheseOptimisers: Muon
using JLD2: jldsave
using Flux

X0_mean_length = 0
deletion_pad = 1.1
per_chain_upper_X0_len = 1 + quantile(Poisson(X0_mean_length), 0.95)

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu

rundir = "runs/branchchain_v1_$(Date(now()))_$(rand(100000:999999))"
mkpath("$(rundir)/samples")

dat = load(PDBSimpleFlat);
#To prevent OOM, we now need to factor in that some low-t samples might have more elements than their X1 lengths:
len_lbs = max.(dat.len, length.(union.(dat.chainids)) .* per_chain_upper_X0_len) .* deletion_pad

model = load_model("condsegment_v1.jld") |> device;

sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.9999f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model)
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

struct BatchDataset{T}
    batchinds::T
end
Base.length(x::BatchDataset) = length(x.batchinds)
Base.getindex(x::BatchDataset, i) = training_prep(x.batchinds[i], dat, deletion_pad, X0_mean_length)
function batchloader(; device=identity, parallel=true)
    uncapped_l2b = length2batch(1500, 1.25)
    batchinds = sample_batched_inds(len_lbs, dat.cluster, l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds)
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss"])
for epoch in 1:7
    if epoch == 6
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5800) 
    end
    for (i, ts) in enumerate(batchloader(; device = device))
        sc_frames = nothing
        if rand() < 0.5
            sc_frames, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks)
        end
        l, grad = Flux.withgradient(model) do m
            frames, aa_logits, count_log, del_logit = m(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, sc_frames = sc_frames)
            l_loc, l_rot, l_aas, l_splits, l_del = losses(BranchChain.P, (frames, aa_logits, count_log, del_logit), ts)
            @show l_loc, l_rot, l_aas, l_splits, l_del
            l_loc + l_rot + l_aas + l_splits + l_del
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        textlog("$(rundir)/log.csv", [epoch, i, sched.lr, l])
        if mod(i, 2000) == 0
            for v in 1:5
                try
                    vidname = "e$(epoch)_b$(i)_samp$(v)"
                    vidpath = vidprepath*vidname
                    x1 = BranchChain.random_template(dat; only_sampled_masked = true)
                    design(model, x1; vidpath = vidpath, device = device)
                catch
                    println("Error in design sample for samp $v")
                end
            end
            jldsave("$(rundir)/model_epoch_$(epoch)_batch_$(i).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
        end
    end
    jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end

