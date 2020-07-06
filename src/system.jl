import LightXML
import Chemfiles

if isdefined(Chemfiles, :atoms)
    Chemfiles_atoms(residue::Chemfiles.Residue) = Chemfiles.atoms(residue)
else
    function Chemfiles_atoms(residue::Chemfiles.Residue)
        count = size(residue)
        result = Array{UInt64}(undef, count)
        Chemfiles.__check(Chemfiles.lib.chfl_residue_atoms(Chemfiles.__const_ptr(residue), pointer(result), count))
        return result
    end
end

include("canonical.jl")

function canonical_mapping(residues, atoms, bonds)
    num_atoms = length(atoms)
    residue_atoms = map((x->x.+1) ∘ Chemfiles_atoms, residues)
    atom_residue = Vector{UInt128}(undef, num_atoms)
    internal_map = Vector{UInt128}(undef, num_atoms)
    for (index, atom_indices) in enumerate(residue_atoms)
        atom_residue[atom_indices] .= index
        internal_map[atom_indices] .= collect(1:length(atom_indices))
    end
    residue_adjacency = map(n->falses(n, n), map(length, residue_atoms))
    for (atom_1, atom_2) in eachcol(bonds)
        index = atom_residue[atom_1]
        if atom_residue[atom_2] == index
            i = internal_map[atom_1]
            j = internal_map[atom_2]
            residue_adjacency[index][i, j] = residue_adjacency[index][j, i] = true
        end
    end
    for index = 1:length(residues)
        canonical_order, canonical_adjacency = canonical_form(residue_adjacency[index])
        residue_atoms[index] .= residue_atoms[index][canonical_order]
        residue_adjacency[index] .= canonical_adjacency
    end
    mapping = Vector{Int}(undef, num_atoms)
    mapping[vcat(residue_atoms...)] .= collect(0:num_atoms-1)
    return residue_atoms, residue_adjacency, mapping
end

function System(file)
    trajectory = Chemfiles.Trajectory(file)
    frame = read(trajectory)
    unit_cell = Chemfiles.UnitCell(frame)
    # cell_lenghts = Chemfiles.lengths(unit_cell)
    # cell_angles = Chemfiles.angles(unit_cell)
    positions = Chemfiles.positions(frame)
    has_velocities = Chemfiles.has_velocities(frame)
    velocities = has_velocities ? Chemfiles.velocities(frame) : zeros(size(positions))
    topology = Chemfiles.Topology(frame)

    num_atoms = size(topology)
    num_residues = Chemfiles.count_residues(topology)

    residues = [Chemfiles.Residue(topology, index) for index = 0:num_residues-1]
    atoms = [Chemfiles.Atom(topology, index) for index = 0:num_atoms-1]
    bonds = Chemfiles.bonds(topology) .+ 1

    residue_atoms, residue_adjacency, mapping = canonical_mapping(residues, atoms, bonds)

    system = Chemfiles.Frame()
    Chemfiles.set_cell!(system, unit_cell)
    has_velocities && Chemfiles.add_velocities!(system)

    for (index, residue) in enumerate(residues)
        new_residue = Chemfiles.Residue(Chemfiles.name(residue), index)
        for i in residue_atoms[index]
            Chemfiles.add_atom!(system, atoms[i], positions[:, i], velocities[:, i])
            Chemfiles.add_atom!(new_residue, mapping[i])
        end
        Chemfiles.add_residue!(system, new_residue)
    end
    bonds = map(x->mapping[x], bonds)
    bond_orders = Chemfiles.bond_orders(topology)
    for (bond, order) in zip(eachcol(bonds), bond_orders)
        Chemfiles.add_bond!(system, bond[1], bond[2], order)
    end
    return system
end
