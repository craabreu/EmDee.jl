export ForceField,
       System

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
    templates::Vector{ResidueTemplate}
    lj₁₋₄::Float64
    coulomb₁₋₄::Float64
end

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

const ATOM_TYPE = LittleDict(:name=>String, :class=>String, :element=>String, :mass=>Float64)

const HARMONIC_BOND = LittleDict(:type1=>String, :type2=>String,
                                 :class1=>String, :class2=>String,
                                 :length=>Float64, :k=>Float64)

const HARMONIC_ANGLE = LittleDict(:type1=>String, :type2=>String, :type3=>String,
                                  :class1=>String, :class2=>String, :class3=>String,
                                  :angle=>Float64, :k=>Float64)

const PERIODIC_TORSION = LittleDict(:type1=>String, :type2=>String, :type3=>String, :type4=>String,
                                    :class1=>String, :class2=>String, :class3=>String, :class4=>String,
                                    :periodicity1=>Int, :phase1=>Float64, :k1=>Float64,
                                    :periodicity2=>Int, :phase2=>Float64, :k2=>Float64,
                                    :periodicity3=>Int, :phase3=>Float64, :k3=>Float64,
                                    :periodicity4=>Int, :phase4=>Float64, :k4=>Float64)

const NONBONDED = LittleDict(:type=>String, :charge=>Float64, :sigma=>Float64, :epsilon=>Float64)

Base.convert(::Type{T}, x::S) where {T<:Number, S<:AbstractString} = parse(T, x)

Base.zero(::Type{String}) = ""

attribute_dicts(list, key) = [LightXML.attributes_dict(a) for e in list for a in e[key]]

function DataFrame(category, element_list, key)
    df = DataFrame((collect∘values)(category), (collect∘keys)(category))
    zeros = LittleDict(string(key)=>zero(value) for (key, value) in category)
    for dict in attribute_dicts(element_list, key)
        append!(df, merge(zeros, dict))
    end
    return df
end

function ForceField(xml_file)
    xroot = LightXML.root(LightXML.parse_file(xml_file))
    atom_types = DataFrame(ATOM_TYPE, xroot["AtomTypes"], "Type")
    type_index = LittleDict(atom_types.name .=> collect(1:nrow(atom_types)))
    templates = Vector{ResidueTemplate}()
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
            natoms = length(atoms) + length(residue_item["ExternalBond"])
            adjmat = falses(natoms, natoms)
            for bond in residue_item["Bond"]
                i, j = [key ∈ ["to", "from"] ? parse(Int, value) + 1 : atom_index[value]
                        for (key, value) in LightXML.attributes_dict(bond)]
                adjmat[i, j] = adjmat[j, i] = true
            end
            for bond in residue_item["ExternalBond"]
                for (key, value) in LightXML.attributes_dict(bond)
                    name = key == "from" ? Chemfiles.name(atoms[parse(Int, value) + 1]) : value
                    push!(atoms, Chemfiles.Atom("$(name)*"))
                    i = atom_index[name]
                    j = length(atoms)
                    adjmat[i, j] = adjmat[j, i] = true
                end
            end
            order, adjmat = canonical_form(adjmat)
            push!(templates, ResidueTemplate(atoms[order], adjmat))
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

    return ForceField(atom_types, bonds, angles, dihedrals, impropers, nonbonded,
                      templates, lj₁₋₄, coulomb₁₋₄)
end

function apply_force_field!(atom_list, adjacency, force_field)
    num_residue_matches = 0
    for (residue_index, matrix) in enumerate(force_field.adjacency)
        if matrix == adjacency
            topology = Chemfiles.Topology(force_field.frame)
            residue = Chemfiles.Residue(topology, residue_index-1)
            residue_atoms = Chemfiles_atoms(residue)
            num_atom_matches = 0
            for (index, atom_index) in enumerate(residue_atoms)
                atom = atom_list[index]
                template = Chemfiles.Atom(topology, atom_index)
                if isapprox(Chemfiles.mass(atom), Chemfiles.mass(template), atol=0.1)
                    num_atom_matches += 1
                end
                Chemfiles.set_property!(atom, "ff.type", Chemfiles.type(template))
                Chemfiles.set_property!(atom, "ff.charge", Chemfiles.charge(template))
            end
            num_atom_matches == length(residue_atoms) && (num_residue_matches += 1)
        end
    end
    return num_residue_matches
end

function pdb_standard_bonds(residues, pdb_names)
    pdb_names_root = LightXML.root(LightXML.parse_file(pdb_names))
    glossary = LittleDict()
    alternatives = LittleDict()
    for element in pdb_names_root["Residue"]
        attributes = LightXML.attributes_dict(element)
        atoms_dict = haskey(attributes, "type") ? copy(glossary[attributes["type"]]) : LittleDict()
        for item in element["Atom"]
            list = join(values(LightXML.attributes_dict(item)), "|")
            atoms_dict[LightXML.attribute(item, "name")] = Regex("\\b($(list))\\b")
        end
        glossary[attributes["name"]] = atoms_dict
        alt_names = []
        while haskey(attributes, "alt$(length(alt_names) + 1)")
            push!(alt_names, attributes["alt$(length(alt_names) + 1)"])
        end
        alternatives[attributes["name"]] = alt_names
    end

    residues_root = LightXML.root(LightXML.parse_file(residues))
    residue_dict = LittleDict{String, Vector{Vector{Regex}}}()
    regex(x) = Regex("\\b($(replace(x, "-" => "_")))\\b")
    for element in residues_root["Residue"]
        attributes = LightXML.attributes_dict(element)
        name = attributes["name"]
        residue_dict[name] = [map(x->get(glossary[name], x, regex(x)),
                              values(LightXML.attributes_dict(item)))
                              for item in element["Bond"]]
        for alt in alternatives[name]
            residue_dict[alt] = residue_dict[name]
        end
    end
    return residue_dict
end

const STANDARD_BONDS = pdb_standard_bonds(broadcast(joinpath, @__DIR__, "data", ["residues.xml", "pdbNames.xml"])...)




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
    # bonds = Chemfiles.bonds(topology) .+ 1
    original_bonds = Chemfiles.bonds(topology) .+ 1
    # residue_atoms = map((x->x.+1) ∘ Chemfiles_atoms, residues)

    bonds = []
    chain_id = ""
    previous_indices = previous_names = []
    for (index, residue) in enumerate(residues)
        atom_indices = Chemfiles_atoms(residue) .+ 1
        atom_names = Chemfiles.name.(atoms[atom_indices])
        residue_name = Chemfiles.name(residue)
        if Chemfiles.property(residue, "is_standard_pdb")
            new_chain = Chemfiles.property(residue, "chainid") != chain_id
            if new_chain
                chain_id = Chemfiles.property(residue, "chainid")
                previous_indices = previous_names = []
            end
            combined_indices = vcat(previous_indices, atom_indices)
            combined_names = vcat(previous_names, atom_names)
            for (atom_1, atom_2) in STANDARD_BONDS[residue_name]
                from = broadcast(occursin, atom_1, combined_names) |> findfirst
                to = broadcast(occursin, atom_2, combined_names) |> findfirst
                from === nothing || to === nothing || push!(bonds, combined_indices[[from, to]])
            end
            previous_indices = atom_indices
            previous_names = "_".*atom_names
        else
            # TODO: For non-standard residues, use bond from original bonds
        end
    end

    # println.(bonds)
    return [Chemfiles.name.(atoms[[i,j]]) for (i,j) in bonds]
    # adjacency, mapping = canonical_mapping!(residue_atoms, bonds)
    # 
    system = Chemfiles.Frame()
    # Chemfiles.set_cell!(system, unit_cell)
    # has_velocities && Chemfiles.add_velocities!(system)
    # for (index, residue) in enumerate(residues)
    #     name = Chemfiles.name(residue)
    #     matches = apply_force_field!(atoms[residue_atoms[index]], adjacency[index], force_field)
    #     if matches != 1
    #         println("$(matches == 0 ? "No" : "Multiple") force field templates for residue $(name)")
    #     end
    #     new_residue = Chemfiles.Residue(name, index)
    #     for i in residue_atoms[index]
    #         Chemfiles.add_atom!(system, atoms[i], positions[:, i], velocities[:, i])
    #         Chemfiles.add_atom!(new_residue, mapping[i])
    #     end
    #     for property in Chemfiles.list_properties(residue)
    #         Chemfiles.set_property!(new_residue, property, Chemfiles.property(residue, property))
    #         @show property Chemfiles.property(residue, property)
    #     end
    #     Chemfiles.add_residue!(system, new_residue)
    # end
    # bond_orders = Chemfiles.bond_orders(topology)
    # for ((atom_1, atom_2), order) in zip(eachcol(bonds), bond_orders)
    #     Chemfiles.add_bond!(system, mapping[atom_1], mapping[atom_2], order)
    # end
    return system
end
