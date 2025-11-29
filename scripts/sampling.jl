using Pkg
Pkg.activate(".")

using BranchChain
using ProtPlot, GLMakie

model = load_model("condsegment_v1.jld");

#pdb_path = "some_local_pdb.pdb"
#to_mask = ["AKAPRSFLYGDDGDFYTESDYFDSW"]
#prot = readpdb(pdb_path);

#Read PDB from online
pdb_path = "9IQP"
to_mask = ["VARGVGVYGMHWFCGEYNFA"] #<- The CDR3

prot = BranchChain.pdbentry(pdb_path)[2:2]; #NOTE <- only taking the second chain in this case.
samp = design(model, X1_from_pdb(prot, to_mask), path = "test.pdb", vidpath = "test_vid") #steps = Float32.([0:0.1:0.9; 0.925:0.025:0.985; 0.99; 0.999; 1.0])

ProtPlot.animate_trajectory_dir(viddir*".mp4", [d for d in readdir(viddir, join=true) if split(d,"/")[end] != ".DS_Store"]|> reverse, color_by=[:chain, :numbering])