export ForceField, System

# TODO list:
# 1. Implement Dissulfide bond assignment
# 2. Check whether graph canonicalization isn't working for nucleic acids with CHARMM

import LightXML
import Chemfiles
using DataFrames
using OrderedCollections

struct ResidueTemplate
    atoms::Vector{Chemfiles.Atom}
    adjacency::BitArray{2}

    function ResidueTemplate(residue, type_masses)
        natoms = length(residue.atoms)
        atom_index = LittleDict(Chemfiles.name.(residue.atoms) .=> 1:natoms)
        adjmat = falses(natoms, natoms)
        for atom_pair in residue.bonds
            i, j = [atom_index[atom] for atom in atom_pair]
            adjmat[i, j] = adjmat[j, i] = true
        end
        atom_masses = [type_masses[Chemfiles.type(atom)] for atom in residue.atoms]
        order, adjmat = canonical_form(adjmat, atom_masses)
        return new(residue.atoms[order], adjmat)
    end
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

const ATOM_TYPE = LittleDict(
    :name => String, :class => String, :element => String, :mass => Float64,
)

const HARMONIC_BOND = LittleDict(
    :type1  => String, :type2  => String,
    :class1 => String, :class2 => String, 
    :length => Float64,
    :k => Float64,
)

const HARMONIC_ANGLE = LittleDict(
    :type1  => String, :type2  => String, :type3  => String,
    :class1 => String, :class2 => String, :class3 => String,
    :angle => Float64,
    :k => Float64,
)

const PERIODIC_TORSION = LittleDict(
    :type1  => String, :type2  => String, :type3  => String, :type4  => String,
    :class1 => String, :class2 => String, :class3 => String, :class4 => String,
    :periodicity1 => Int, :phase1 => Float64, :k1 => Float64,
    :periodicity2 => Int, :phase2 => Float64, :k2 => Float64,
    :periodicity3 => Int, :phase3 => Float64, :k3 => Float64,
    :periodicity4 => Int, :phase4 => Float64, :k4 => Float64,
    :periodicity5 => Int, :phase5 => Float64, :k5 => Float64,
    :periodicity6 => Int, :phase6 => Float64, :k6 => Float64,
)

const NONBONDED = LittleDict(
    :type => String, :charge => Float64, :sigma => Float64, :epsilon => Float64,
)

mutable struct Residue
    atoms
    bonds
    external_bonds
    Residue() = new([], [], [])
    Residue(atoms, bonds, external_bonds) = new(atoms, bonds, external_bonds)
end

Base.copy(r::Residue) = Residue(copy(r.atoms), copy(r.bonds), copy(r.external_bonds))

sanitized(str) = replace(replace(replace(str, "-" => "_"), "'" => "p"), "*" => "a")

function AddAtom!(residue, attributes)
    atom = Chemfiles.Atom(sanitized(attributes["name"]))
    charge = convert(Float64, get(attributes, "charge", 0))
    Chemfiles.set_charge!(atom, charge)
    Chemfiles.set_type!(atom, attributes["type"])
    push!(residue.atoms, atom)
end

function AddBond!(residue, attributes)
    push!(residue.bonds, Set(sanitized.(values(attributes))))
end

function AddExternalBond!(residue, attributes)
    push!(residue.external_bonds, sanitized(attributes["atomName"]))
end

function ChangeAtom!(residue, attributes)
    name = sanitized(attributes["name"])
    for atom in residue.atoms
        if Chemfiles.name(atom) == name
            charge = convert(Float64, get(attributes, "charge", 0))
            Chemfiles.set_charge!(atom, charge)
            Chemfiles.set_type!(atom, attributes["type"])
            return nothing
        end
    end
end

function RemoveAtom!(residue, attributes)
    atom = sanitized(attributes["name"])
    residue.atoms = filter(x -> Chemfiles.name(x) ≠ atom, residue.atoms)
end

function RemoveBond!(residue, attributes)
    bond = Set(sanitized.(attributes[a] for a in ["atomName1", "atomName2"]))
    residue.bonds = filter(x -> x ≠ bond, residue.bonds)
end

function RemoveExternalBond!(residue, attributes)
    atom = sanitized(attributes["atomName"])
    residue.external_bonds = filter(x -> x ≠ atom, residue.external_bonds)
end

Base.convert(::Type{T}, x::S) where {T<:Number,S<:AbstractString} = parse(T, x)

Base.zero(::Type{String}) = ""

function DataFrame(category, element_list, key)
    df = DataFrame((collect ∘ values)(category), (collect ∘ keys)(category))
    zeros = LittleDict(string(key) => zero(value) for (key, value) in category)
    attribute_dicts(list, key) = [LightXML.attributes_dict(a) for e in list for a in e[key]]
    for dict in attribute_dicts(element_list, key)
        append!(df, merge(zeros, dict))
    end
    # TODO: remove all empty columns
    return df
end

function ForceField(xml_file)
    xroot = LightXML.root(LightXML.parse_file(xml_file))
    patches = LittleDict(
        LightXML.attribute(item, "name") => [
            Symbol(LightXML.name(child)*"!") => LightXML.attributes_dict(child)
            for child in LightXML.child_elements(item)
        ]
        for elem in xroot["Patches"] for item in elem["Patch"]
    )
    atom_types = DataFrame(ATOM_TYPE, xroot["AtomTypes"], "Type")
    type_index = LittleDict(atom_types.name .=> collect(1:nrow(atom_types)))
    type_masses = LittleDict(atom_types.name .=> atom_types.mass)
    templates = OrderedDict{String,ResidueTemplate}()
    for elem in xroot["Residues"]
        for (residue_index, residue_item) in enumerate(elem["Residue"])
            residue = Residue()
            names = []
            for (index, atom_item) in enumerate(residue_item["Atom"])
                attributes = LightXML.attributes_dict(atom_item)
                push!(names, attributes["name"])
                AddAtom!(residue, attributes)
            end
            for bond in residue_item["Bond"]
                atom_names = [
                    key ∈ ["to", "from"] ? names[parse(Int, value)+1] : value
                    for (key, value) in LightXML.attributes_dict(bond)
                ]
                AddBond!(residue, Dict(["atomName1", "atomName2"] .=> atom_names))
            end
            for bond in residue_item["ExternalBond"]
                attributes = LightXML.attributes_dict(bond)
                if haskey(attributes, "from")
                    attributes["atomName"] = names[parse(Int, attributes["from"])+1]
                end
                AddExternalBond!(residue, attributes)
            end
            residue_name = LightXML.attribute(residue_item, "name")
            templates[residue_name] = ResidueTemplate(residue, type_masses)
            for item in residue_item["AllowPatch"]
                patch = LightXML.attribute(item, "name")
                patched_residue = copy(residue)
                for (action, attributes) in patches[patch]
                    getfield(@__MODULE__, action)(patched_residue, attributes)
                end
                templates["$residue_name($patch)"] = ResidueTemplate(patched_residue, type_masses)
            end
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
    return ForceField(atom_types, bonds, angles, dihedrals, impropers,
                      nonbonded, templates, lj₁₋₄, coulomb₁₋₄)
end

function pdb_aliases(pdb_aliases_file)
    name(element) = LightXML.attribute(element, "name")
    mass(element) = parse(Float64, LightXML.attribute(element, "mass"))
    regex(item) = Regex(LightXML.attribute(item, "code"))
    ids(item) = parse.(Int, values(LightXML.attributes_dict(item)))
    xroot = LightXML.root(LightXML.parse_file(pdb_aliases_file))
    masses = LittleDict(name(e) => mass(e) for e in xroot["Elements"][1]["Element"])
    regex_codes = regex.(xroot["RegularExpressions"][1]["Regex"])
    std_bonds = LittleDict(name(e) => ids.(e["Bond"]) for e in xroot["Residue"])
    return masses, regex_codes, std_bonds
end

const PDB_MASSES, PDB_REGEX_CODES, PDB_STD_BONDS =
    pdb_aliases(joinpath(@__DIR__, "data", "pdb_aliases.xml"))

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
    Chemfiles.set_name!.(atoms, (sanitized∘Chemfiles.name).(atoms))
    residues = [Chemfiles.Residue(topology, index) for index = 0:num_residues-1]
    residue_atoms = map((x -> x .+ 1) ∘ Chemfiles_atoms, residues)
    atom_residues = Vector{Int}(undef, num_atoms)
    internal_map = Vector{Int}(undef, num_atoms)
    for (index, atoms_list) in enumerate(residue_atoms)
        atom_residues[atoms_list] .= index
        internal_map[atoms_list] .= collect(1:length(atoms_list))
    end

    is_std_pdb = Chemfiles.property.(residues, "is_standard_pdb")
    for (atom, residue_index) in zip(atoms, atom_residues)
        if is_std_pdb[residue_index]
            element = match(r"[HCNOPS]", Chemfiles.type(atom)).match
            Chemfiles.set_mass!(atom, PDB_MASSES[element])
        end
    end

    bonds = [
        collect(bond)
        for bond in eachcol(Chemfiles.bonds(topology) .+ 1)
        if !all(is_std_pdb[atom_residues[bond]])
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
        index = atom_residues[atom_1]
        if atom_residues[atom_2] == index
            i, j = internal_map[[atom_1, atom_2]]
            adjacency_matrices[index][i, j] = adjacency_matrices[index][j, i] = true
        end
    end

    for (residue, indices, matrix) in zip(residues, residue_atoms, adjacency_matrices)
        atom_masses = Chemfiles.mass.(atoms[indices])
        canonical_order, canonical_matrix = canonical_form(matrix, atom_masses)
        matches = [n for (n, t) in force_field.templates if t.adjacency == canonical_matrix]
        name = Chemfiles.name(residue)
        # println(Chemfiles.name.(atoms[indices][canonical_order]))
        # println(Chemfiles.name.(force_field.templates["ADE"].atoms))
        # println.(size.([canonical_matrix, force_field.templates["ADE"].adjacency]))
        # println(canonical_matrix)
        # println(force_field.templates["ADE"].adjacency)
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
    location = Vector{Int}(undef, num_atoms)
    atom_index = 0
    for (index, residue) in enumerate(residues)
        new_residue = Chemfiles.Residue(Chemfiles.name(residue), index)
        for i in residue_atoms[index]
            Chemfiles.add_atom!(system, atoms[i], positions[:, i], velocities[:, i])
            Chemfiles.add_atom!(new_residue, atom_index)
            location[i] = atom_index
            atom_index += 1
        end
        Chemfiles.add_residue!(system, new_residue)
    end
    for bond in bonds
        Chemfiles.add_bond!(system, location[bond]...)
    end
    return system
end
