export ForceField,
       System

import LightXML
import Chemfiles
using DataFrames
using OrderedCollections

struct ForceField
    atom_types::DataFrame
    bond_types::DataFrame
    angle_types::DataFrame
    dihedral_types::DataFrame
    improper_types::DataFrame
    nonbonded::DataFrame
    factors::DataFrame
    from_residue::Vector{String}
    frame::Chemfiles.Frame
    adjacency::Vector{BitArray{2}}
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
    df = DataFrame(collect(values(category)), collect(keys(category)))
    zeros = LittleDict(string(key)=>zero(value) for (key, value) in category)
    for dict in attribute_dicts(element_list, key)
        append!(df, merge(zeros, dict))
    end
    return df
end

function ForceField(xml_file)
    xroot = LightXML.root(LightXML.parse_file(xml_file))
    atom_types = DataFrame(ATOM_TYPE, xroot["AtomTypes"], "Type")
    in_types = LittleDict(atom_types.name .=> collect(1:nrow(atom_types)))
    index = 0
    atoms = Vector{Chemfiles.Atom}()
    residues = Vector{Chemfiles.Residue}()
    residue_atoms = Vector{Vector{Int}}()
    bond_atom_1 = Vector{Int}()
    bond_atom_2 = Vector{Int}()
    for elem in xroot["Residues"]
        for (residue_index, residue_item) in enumerate(elem["Residue"])
            residue_name = LightXML.attribute(residue_item, "name")
            residue = Chemfiles.Residue(residue_name, residue_index)
            by_name = LittleDict()
            name_of = LittleDict()
            atoms_in_residue = Vector{Int}()
            for (atom_index, atom_item) in enumerate(residue_item["Atom"])
                attributes = LightXML.attributes_dict(atom_item)
                name = attributes["name"]
                type = attributes["type"]
                charge = convert(Float64, get(attributes, "charge", 0))
                atom = Chemfiles.Atom(name)
                Chemfiles.set_type!(atom, type)
                Chemfiles.set_charge!(atom, charge)
                Chemfiles.set_mass!(atom, atom_types[in_types[type],:].mass)
                index += 1
                by_name[name] = index
                name_of[string(atom_index-1)] = name
                push!(atoms_in_residue, index)
                push!(atoms, atom)
            end
            push!(residues, residue)
            push!(residue_atoms, atoms_in_residue)
            for bond in residue_item["Bond"]
                for (key, value) in LightXML.attributes_dict(bond)
                    if key ∈ ["atomName1", "from"]
                        push!(bond_atom_1, by_name[key == "atomName1" ? value : name_of[value]])
                    elseif key ∈ ["atomName2", "to"]
                        push!(bond_atom_2, by_name[key == "atomName2" ? value : name_of[value]])
                    end
                end
            end
            external_bonds = []
            for bond in residue_item["ExternalBond"]
                for (key, value) in LightXML.attributes_dict(bond)
                    push!(external_bonds, key == "atomName" ? name_of[value] : value)
                end
            end
            Chemfiles.set_property!(residue, "external_bonds", join(external_bonds, ";"))
        end
    end
    bonds = vcat(bond_atom_1', bond_atom_2')
    adjacency, mapping = canonical_mapping!(residue_atoms, bonds)
    frame = Chemfiles.Frame()
    for (residue, atoms_in_residue) in zip(residues, residue_atoms)
        for index in atoms_in_residue
            atom = atoms[index]
            Chemfiles.add_atom!(frame, atom, zeros(3))
            Chemfiles.add_atom!(residue, index-1)
        end
        Chemfiles.add_residue!(frame, residue)
    end
    for (atom_1, atom_2) in eachcol(bonds)
        Chemfiles.add_bond!(frame, mapping[atom_1], mapping[atom_2])
    end

    bonds = DataFrame(HARMONIC_BOND, xroot["HarmonicBondForce"], "Bond")
    angles = DataFrame(HARMONIC_ANGLE, xroot["HarmonicAngleForce"], "Angle")
    dihedrals = DataFrame(PERIODIC_TORSION, xroot["PeriodicTorsionForce"], "Proper")
    impropers = DataFrame(PERIODIC_TORSION, xroot["PeriodicTorsionForce"], "Improper")
    nonbonded = DataFrame(NONBONDED, xroot["NonbondedForce"], "Atom")

    dicts = attribute_dicts(xroot["NonbondedForce"], "UseAttributeFromResidue")
    from_residue = (collect∘values∘merge)(dicts...)

    factors = DataFrame([Float64, Float64], [:lj14scale, :coulomb14scale])
    defaults = LittleDict("lj14scale"=>1.0, "coulomb14scale"=>1.0)
    append!(factors, merge(defaults, LightXML.attributes_dict(xroot["NonbondedForce"][1])))

    return ForceField(atom_types, bonds, angles, dihedrals, impropers, nonbonded,
                      factors, from_residue, frame, adjacency)
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
        name = Chemfiles.name(residue)
        matches = apply_force_field!(atoms[residue_atoms[index]], adjacency[index], force_field)
        if matches != 1
            error("$(matches == 0 ? "No" : "Multiple") force field templates for residue $(name)")
        end
        new_residue = Chemfiles.Residue(name, index)
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
