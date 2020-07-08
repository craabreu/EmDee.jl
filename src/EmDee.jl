module EmDee

include("vec3.jl")
include("lennard_jones.jl")
include("nonbonded.jl")
include("canonical.jl")
include("force_field.jl")
include("system.jl")

xml_file = "/home/charlles/Projects/EmDee.jl/test/data/dibenzo-p-dioxin-in-water.xml"
pdb_file = "/home/charlles/Projects/EmDee.jl/test/data/dibenzo-p-dioxin-in-water.pdb"

force_field = ForceField(xml_file)

test_file = "/home/charlles/Projects/EmDee.jl/test/test.pdb"

system = System(pdb_file, force_field)
traj = Chemfiles.Trajectory(test_file, 'w')
Chemfiles.write(traj, system)
close(traj)

end
