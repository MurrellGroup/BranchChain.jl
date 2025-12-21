using Pkg
Pkg.activate(temp=true)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.add(["CUDA", "cuDNN"])

ENV["CUDA_VISIBLE_DEVICES"] = 0
using CUDA, cuDNN
device!(0) # <- pick your GPU (zero indexed)

using BranchChain
dev = BranchChain.gpu

model = load_model("branchchain_tune1.jld") |> dev;

template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence] #Redesign the whole second chain
#to_redesign = ["QVQLQESGGGLVQAGGSLRLSCAASGSDFSSSTMGWYRQAPGKQREFVAISSEGSTSYAGSVKGRFTISRDNAKNTVYLQMNSLEPED","QVTVSA"] #Redesign part of the second chain
samp = design(model, X1_from_pdb(template, to_redesign), steps = 200, path = "samp.pdb", device = dev, vidpath = "samp-anim");
