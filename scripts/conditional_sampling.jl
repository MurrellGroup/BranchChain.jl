using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

ENV["CUDA_VISIBLE_DEVICES"] = 1
using BranchChain, CUDA, DLProteinFormats

device!(0) #<-pick your GPU (zero indexed)
devi = BranchChain.gpu

model = model = load_model("branchchain_featuretune1.jld") |> devi;

feature_table = DLProteinFormats.load(PDBTable);
sampling_ff = featurizer(feature_table, DLProteinFormats.CHAIN_FEATS_V1)

#Design a single chain RBD binder with nanobody-like features:
feature_table[feature_table.pdb_id .== "7F5H",:]
template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence]
samp = design(model, X1_from_pdb(template, to_redesign), template.name, [t.id for t in template], sampling_ff;
        device = devi, recycles = 1, steps = 200, vidpath = "testfeat_vid", path = "testfeat.pdb")