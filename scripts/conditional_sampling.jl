using Pkg
Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.activate(temp=true)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.add(["DLProteinFormats", "CUDA", "cuDNN"])

ENV["CUDA_VISIBLE_DEVICES"] = 0
using CUDA, cuDNN
device!(0) # <- pick your GPU (zero indexed)

using BranchChain
dev = BranchChain.gpu

model = model = load_model("branchchain_featuretune1.jld") |> dev;

using DLProteinFormats

feature_table = DLProteinFormats.load(PDBTable);
sampling_ff = featurizer(feature_table, DLProteinFormats.CHAIN_FEATS_V1)

# Design a single chain RBD binder with nanobody-like features:
feature_table[feature_table.pdb_id .== "7F5H", :]
template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence]
samp = design(model, X1_from_pdb(template, to_redesign), template.name, [t.id for t in template], sampling_ff;
        device = dev, recycles = 1, steps = 200, path = "samp-feat.pdb", vidpath = "samp-feat-anim")
