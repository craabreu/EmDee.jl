import LightXML
import Chemfiles
using DataFrames

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

function dataframe(category, element_list, key)
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
    atom_types = dataframe(ATOM_TYPE, xroot["AtomTypes"], "Type")
    bond_types = dataframe(HARMONIC_BOND, xroot["HarmonicBondForce"], "Bond")
    angle_types = dataframe(HARMONIC_ANGLE, xroot["HarmonicAngleForce"], "Angle")
    dihedral_types = dataframe(PERIODIC_TORSION, xroot["PeriodicTorsionForce"], "Proper")
    improper_types = dataframe(PERIODIC_TORSION, xroot["PeriodicTorsionForce"], "Improper")
    nonbonded = dataframe(NONBONDED, xroot["NonbondedForce"], "Atom")

    index = 0
    in_types = Dict(atom_types.name .=> collect(1:nrow(atom_types)))
    in_nonbonded = Dict(nonbonded.type .=> collect(1:nrow(nonbonded)))
    in_frame = Dict()
    residues = []
    atoms = []
    residue_atoms = Vector{Vector{Int}}()
    bond_atom_1 = Vector{Int}()
    bond_atom_2 = Vector{Int}()
    for elem in xroot["Residues"]
        for (residue_index, residue_item) in enumerate(elem["Residue"])
            residue = Chemfiles.Residue(LightXML.attribute(residue_item, "name"), residue_index)
            push!(residues, residue)
            atoms_in_residue = []
            for (atom_index, atom_item) in enumerate(residue_item["Atom"])
                name = LightXML.attribute(atom_item, "name")
                type = LightXML.attribute(atom_item, "type")
                charge = parse(Float64, LightXML.attribute(atom_item, "charge"))
                atom = Chemfiles.Atom(name)
                data = atom_types[in_types[type],:]
                parameters = nonbonded[in_nonbonded[type],:]
                Chemfiles.set_type!(atom, type)
                Chemfiles.set_charge!(atom, charge)
                Chemfiles.set_mass!(atom, data.mass)
                Chemfiles.set_property!(atom, "class", data.class)
                Chemfiles.set_property!(atom, "element", data.element)
                Chemfiles.set_property!(atom, "sigma", parameters.sigma)
                Chemfiles.set_property!(atom, "epsilon", parameters.epsilon)
                index += 1
                in_frame[name] = index
                push!(atoms_in_residue, index)
                push!(atoms, atom)
            end
            push!(residue_atoms, atoms_in_residue)
            for bond in residue_item["Bond"]
                push!(bond_atom_1, in_frame[LightXML.attribute(bond, "atomName1")])
                push!(bond_atom_2, in_frame[LightXML.attribute(bond, "atomName2")])
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
    return ForceField(atom_types, bond_types, angle_types, dihedral_types, improper_types,
                      frame, adjacency)
end
