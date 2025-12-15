using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

ENV["CUDA_VISIBLE_DEVICES"] = 0
using BranchChain, CUDA

device!(0) #<-pick your GPU (zero indexed)
devi = BranchChain.gpu

model = load_model("branchchain_tune1.jld") |> devi;

template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence] #Redesign the whole second chain
#to_redesign = ["QVQLQESGGGLVQAGGSLRLSCAASGSDFSSSTMGWYRQAPGKQREFVAISSEGSTSYAGSVKGRFTISRDNAKNTVYLQMNSLEPED","QVTVSA"] #Redesign part of the second chain
samp = design(model, X1_from_pdb(template, to_redesign), steps = 200, path = "test.pdb", device = devi, vidpath = "test_anim");
