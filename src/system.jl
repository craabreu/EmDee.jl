export System

import Chemfiles

function System(file, force_field)
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
    residue_atoms = map((x->x.+1) ∘ Chemfiles_atoms, residues)

    adjacency, mapping = canonical_mapping!(residue_atoms, bonds)

    system = Chemfiles.Frame()
    Chemfiles.set_cell!(system, unit_cell)
    has_velocities && Chemfiles.add_velocities!(system)
    for (index, residue) in enumerate(residues)
        matched = apply_force_field!(atoms[residue_atoms[index]], adjacency[index], force_field)
        matched || error("Incompatible force field")
        new_residue = Chemfiles.Residue(Chemfiles.name(residue), index)
        for i in residue_atoms[index]
            Chemfiles.add_atom!(system, atoms[i], positions[:, i], velocities[:, i])
            Chemfiles.add_atom!(new_residue, mapping[i])
        end
        Chemfiles.add_residue!(system, new_residue)
    end
    bond_orders = Chemfiles.bond_orders(topology)
    for ((atom_1, atom_2), order) in zip(eachcol(bonds), bond_orders)
        Chemfiles.add_bond!(system, mapping[atom_1], mapping[atom_2], order)
    end
    return system
end
