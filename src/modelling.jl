export ForceField, System

import LightXML
import Chemfiles
using DataFrames
using OrderedCollections

struct ResidueTemplate
    atoms::Vector{Chemfiles.Atom}
    adjacency::BitArray{2}
end

struct ForceField
    atom_types::DataFrame
    bond_types::DataFrame
    angle_types::DataFrame
    dihedral_types::DataFrame
    improper_types::DataFrame
    nonbonded::DataFrame
    templates::OrderedDict{String,ResidueTemplate}
    lj₁₋₄::Float64
    coulomb₁₋₄::Float64
end

if isdefined(Chemfiles, :atoms)
    Chemfiles_atoms(residue::Chemfiles.Residue) = Chemfiles.atoms(residue)
else
    function Chemfiles_atoms(residue::Chemfiles.Residue)
        count = size(residue)
        result = Array{UInt64}(undef, count)
        Chemfiles.__check(Chemfiles.lib.chfl_residue_atoms(
            Chemfiles.__const_ptr(residue),
            pointer(result),
            count,
        ))
        return result
    end
end

const ATOM_TYPE =
    LittleDict(:name => String, :class => String, :element => String, :mass => Float64)

const HARMONIC_BOND = LittleDict(
    :type1 => String,
    :type2 => String,
    :class1 => String,
    :class2 => String,
    :length => Float64,
    :k => Float64,
)

const HARMONIC_ANGLE = LittleDict(
    :type1 => String,
    :type2 => String,
    :type3 => String,
    :class1 => String,
    :class2 => String,
    :class3 => String,
    :angle => Float64,
    :k => Float64,
)

const PERIODIC_TORSION = LittleDict(
    :type1 => String,
    :type2 => String,
    :type3 => String,
    :type4 => String,
    :class1 => String,
    :class2 => String,
    :class3 => String,
    :class4 => String,
    :periodicity1 => Int,
    :phase1 => Float64,
    :k1 => Float64,
    :periodicity2 => Int,
    :phase2 => Float64,
    :k2 => Float64,
    :periodicity3 => Int,
    :phase3 => Float64,
    :k3 => Float64,
    :periodicity4 => Int,
    :phase4 => Float64,
    :k4 => Float64,
)

const NONBONDED =
    LittleDict(:type => String, :charge => Float64, :sigma => Float64, :epsilon => Float64)

Base.convert(::Type{T}, x::S) where {T<:Number,S<:AbstractString} = parse(T, x)

Base.zero(::Type{String}) = ""

function DataFrame(category, element_list, key)
    df = DataFrame((collect ∘ values)(category), (collect ∘ keys)(category))
    zeros = LittleDict(string(key) => zero(value) for (key, value) in category)
    attribute_dicts(list, key) = [LightXML.attributes_dict(a) for e in list for a in e[key]]
    for dict in attribute_dicts(element_list, key)
        append!(df, merge(zeros, dict))
    end
    return df
end

function ForceField(xml_file)
    xroot = LightXML.root(LightXML.parse_file(xml_file))
    atom_types = DataFrame(ATOM_TYPE, xroot["AtomTypes"], "Type")
    type_index = LittleDict(atom_types.name .=> collect(1:nrow(atom_types)))
    templates = OrderedDict{String,ResidueTemplate}()
    for elem in xroot["Residues"]
        for (residue_index, residue_item) in enumerate(elem["Residue"])
            atoms = []
            atom_index = LittleDict()
            for (index, atom_item) in enumerate(residue_item["Atom"])
                attributes = LightXML.attributes_dict(atom_item)
                name = attributes["name"]
                type = attributes["type"]
                charge = convert(Float64, get(attributes, "charge", 0))
                atom = Chemfiles.Atom(name)
                Chemfiles.set_type!(atom, type)
                Chemfiles.set_charge!(atom, charge)
                Chemfiles.set_mass!(atom, atom_types[type_index[type], :].mass)
                push!(atoms, atom)
                atom_index[name] = index
            end
            natoms = length(atoms) #+ length(residue_item["ExternalBond"])
            adjmat = falses(natoms, natoms)
            for bond in residue_item["Bond"]
                i, j = [
                    key ∈ ["to", "from"] ? parse(Int, value) + 1 : atom_index[value]
                    for (key, value) in LightXML.attributes_dict(bond)
                ]
                adjmat[i, j] = adjmat[j, i] = true
            end
            # for bond in residue_item["ExternalBond"]
            #     for (key, value) in LightXML.attributes_dict(bond)
            #         name = key == "from" ? Chemfiles.name(atoms[parse(Int, value) + 1]) : value
            #         push!(atoms, Chemfiles.Atom("$(name)*"))
            #         i = atom_index[name]
            #         j = length(atoms)
            #         adjmat[i, j] = adjmat[j, i] = true
            #     end
            # end
            order, adjmat = canonical_form(adjmat, Chemfiles.mass.(atoms))
            residue_name = LightXML.attribute(residue_item, "name")
            templates[residue_name] = ResidueTemplate(atoms[order], adjmat)
        end
    end

    bonds = DataFrame(HARMONIC_BOND, xroot["HarmonicBondForce"], "Bond")
    angles = DataFrame(HARMONIC_ANGLE, xroot["HarmonicAngleForce"], "Angle")
    dihedrals = DataFrame(PERIODIC_TORSION, xroot["PeriodicTorsionForce"], "Proper")
    impropers = DataFrame(PERIODIC_TORSION, xroot["PeriodicTorsionForce"], "Improper")
    nonbonded = DataFrame(NONBONDED, xroot["NonbondedForce"], "Atom")

    scaling_factors = LightXML.attributes_dict(xroot["NonbondedForce"][1])
    lj₁₋₄ = get(scaling_factors, "lj14scale", 1.0)
    coulomb₁₋₄ = get(scaling_factors, "coulomb14scale", 1.0)

    return ForceField(
        atom_types,
        bonds,
        angles,
        dihedrals,
        impropers,
        nonbonded,
        templates,
        lj₁₋₄,
        coulomb₁₋₄,
    )
end

function pdb_aliases(pdb_aliases_file)
    name(element) = LightXML.attribute(element, "name")
    regex(item) = Regex(LightXML.attribute(item, "code"))
    ids(item) = parse.(Int, values(LightXML.attributes_dict(item)))
    xroot = LightXML.root(LightXML.parse_file(pdb_aliases_file))
    regex_codes = regex.(xroot["RegularExpressions"][1]["Regex"])
    std_bonds = LittleDict(name(e) => ids.(e["Bond"]) for e in xroot["Residue"])
    return regex_codes, std_bonds
end

const PDB_REGEX_CODES, PDB_STD_BONDS = pdb_aliases(joinpath(@__DIR__, "data", "pdb_aliases.xml"))

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
    atoms = [Chemfiles.Atom(topology, index) for index = 0:num_atoms-1]
    residues = [Chemfiles.Residue(topology, index) for index = 0:num_residues-1]
    residue_atoms = map((x -> x .+ 1) ∘ Chemfiles_atoms, residues)
    atom_residue = Vector{Int}(undef, num_atoms)
    internal_map = Vector{Int}(undef, num_atoms)
    for (index, atoms_list) in enumerate(residue_atoms)
        atom_residue[atoms_list] .= index
        internal_map[atoms_list] .= collect(1:length(atoms_list))
    end

    is_std_pdb = Chemfiles.property.(residues, "is_standard_pdb")
    bonds = [
        collect(bond)
        for
        bond in eachcol(Chemfiles.bonds(topology) .+ 1) if !all(is_std_pdb[atom_residue[bond]])
    ]
    chain_id = ""
    previous_indices = previous_names = []
    for index = 1:num_residues
        residue = residues[index]
        atom_indices = residue_atoms[index]
        atom_names = Chemfiles.name.(atoms[atom_indices])
        residue_name = Chemfiles.name(residue)
        if is_std_pdb[index]
            new_chain = Chemfiles.property(residue, "chainid") != chain_id
            if new_chain
                chain_id = Chemfiles.property(residue, "chainid")
                previous_indices = previous_names = []
            end
            combined_indices = vcat(previous_indices, atom_indices)
            combined_names = vcat(previous_names, atom_names)
            for (atom_1, atom_2) in PDB_STD_BONDS[residue_name]
                from = broadcast(occursin, PDB_REGEX_CODES[atom_1], combined_names) |> findfirst
                to = broadcast(occursin, PDB_REGEX_CODES[atom_2], combined_names) |> findfirst
                from === nothing || to === nothing || push!(bonds, combined_indices[[from, to]])
            end
            previous_indices = atom_indices
            previous_names = "_" .* atom_names
        end
    end

    adjacency_matrices = map(n -> falses(n, n), length.(residue_atoms))
    for (atom_1, atom_2) in bonds
        index = atom_residue[atom_1]
        if atom_residue[atom_2] == index
            i = internal_map[atom_1]
            j = internal_map[atom_2]
            adjacency_matrices[index][i, j] = adjacency_matrices[index][j, i] = true
        end
    end

    for (residue, indices, matrix) in zip(residues, residue_atoms, adjacency_matrices)
        atom_masses = Chemfiles.mass.(atoms[indices])
        canonical_order, canonical_matrix = canonical_form(matrix, atom_masses)
        matches = [n for (n, t) in force_field.templates if t.adjacency == canonical_matrix]
        name = Chemfiles.name(residue)
        length(matches) == 0 &&
            error("No force field templates matched residue $(name)")
        length(matches) > 1 &&
            error("Multiple force field templates $(matches) matched residue $(name)")
        template_atoms = force_field.templates[matches[1]].atoms
        for (i, a) in zip(indices[canonical_order], template_atoms)
            Chemfiles.set_property!(atoms[i], "ff.type", Chemfiles.type(a))
            Chemfiles.set_property!(atoms[i], "ff.charge", Chemfiles.charge(a))
        end
    end

    system = Chemfiles.Frame()
    Chemfiles.set_cell!(system, unit_cell)
    has_velocities && Chemfiles.add_velocities!(system)
    position = Vector{Int}(undef, num_atoms)
    atom_index = 0
    for (index, residue) in enumerate(residues)
        new_residue = Chemfiles.Residue(Chemfiles.name(residue), index)
        for i in residue_atoms[index]
            Chemfiles.add_atom!(system, atoms[i], positions[:, i], velocities[:, i])
            Chemfiles.add_atom!(new_residue, atom_index)
            position[i] = atom_index
            atom_index += 1
        end
        Chemfiles.add_residue!(system, new_residue)
    end
    for bond in bonds
        Chemfiles.add_bond!(system, position[bond]...)
    end
    return system
end
