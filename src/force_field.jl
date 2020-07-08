export ForceField

import LightXML
import Chemfiles
using DataFrames

struct ForceField
    atom_types::DataFrame
    bond_types::DataFrame
    angle_types::DataFrame
    dihedral_types::DataFrame
    improper_types::DataFrame
    nonbonded::DataFrame
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

const ATOM_TYPE = Dict(:name=>String, :class=>String, :element=>String, :mass=>Float64)

const HARMONIC_BOND = Dict(:type1=>String, :type2=>String, :length=>Float64, :k=>Float64)

const HARMONIC_ANGLE = Dict(:type1=>String, :type2=>String, :type3=>String,
                            :angle=>Float64, :k=>Float64)

const PERIODIC_TORSION = Dict(:type1=>String, :type2=>String, :type3=>String, :type4=>String,
                              :periodicity1=>Int, :phase1=>Float64, :k1=>Float64,
                              :periodicity2=>Int, :phase2=>Float64, :k2=>Float64,
                              :periodicity3=>Int, :phase3=>Float64, :k3=>Float64)

const NONBONDED = Dict(:type=>String, :sigma=>Float64, :epsilon=>Float64)

Base.convert(::Type{T}, x::S) where {T<:Number, S<:AbstractString} = parse(T, x)

Base.zero(::Type{String}) = ""

function DataFrame(category, element_list, key)
    df = DataFrame(collect(values(category)), collect(keys(category)))
    zeros = Dict(string(key)=>zero(value) for (key, value) in category)
    for element in element_list
        for item in element[key]
            append!(df, merge(zeros, LightXML.attributes_dict(item)))
        end
    end
    return df
end

function ForceField(xml_file)
    xroot = LightXML.root(LightXML.parse_file(xml_file))
    atom_types = DataFrame(ATOM_TYPE, xroot["AtomTypes"], "Type")
    in_types = Dict(atom_types.name .=> collect(1:nrow(atom_types)))
    index = 0
    atoms = Vector{Chemfiles.Atom}()
    residues = Vector{Chemfiles.Residue}()
    residue_atoms = Vector{Vector{Int}}()
    bond_atom_1 = Vector{Int}()
    bond_atom_2 = Vector{Int}()
    for elem in xroot["Residues"]
        for (residue_index, residue_item) in enumerate(elem["Residue"])
            residue = Chemfiles.Residue(LightXML.attribute(residue_item, "name"), residue_index)
            in_residue = Dict()
            atoms_in_residue = Vector{Int}()
            for (atom_index, atom_item) in enumerate(residue_item["Atom"])
                name = LightXML.attribute(atom_item, "name")
                type = LightXML.attribute(atom_item, "type")
                charge = parse(Float64, LightXML.attribute(atom_item, "charge"))
                atom = Chemfiles.Atom(name)
                Chemfiles.set_type!(atom, type)
                Chemfiles.set_charge!(atom, charge)
                Chemfiles.set_mass!(atom, atom_types[in_types[type],:].mass)
                in_residue[name] = (index += 1)
                push!(atoms_in_residue, index)
                push!(atoms, atom)
            end
            push!(residues, residue)
            push!(residue_atoms, atoms_in_residue)
            for bond in residue_item["Bond"]
                push!(bond_atom_1, in_residue[LightXML.attribute(bond, "atomName1")])
                push!(bond_atom_2, in_residue[LightXML.attribute(bond, "atomName2")])
            end
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
    return ForceField(atom_types, bonds, angles, dihedrals, impropers, nonbonded, frame, adjacency)
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