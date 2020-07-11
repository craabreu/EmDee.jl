#!/usr/bin/env julia

using LightXML
using OrderedCollections

function pdb_standard_bonds(residues, pdb_names)
    pdb_names_root = root(parse_file(pdb_names))
    glossary = OrderedDict()
    alternatives = OrderedDict()
    for element in pdb_names_root["Residue"]
        attributes = attributes_dict(element)
        atoms_dict = haskey(attributes, "type") ? copy(glossary[attributes["type"]]) : OrderedDict()
        for item in element["Atom"]
            list = join(values(attributes_dict(item)), "|")
            atoms_dict[attribute(item, "name")] = "\\b($(list))\\b"
        end
        glossary[attributes["name"]] = atoms_dict
        alt_names = []
        while haskey(attributes, "alt$(length(alt_names) + 1)")
            push!(alt_names, attributes["alt$(length(alt_names) + 1)"])
        end
        alternatives[attributes["name"]] = alt_names
    end

    residues_root = root(parse_file(residues))
    residue_dict = OrderedDict{String, Vector{Vector{String}}}()
    regex(x) = "\\b($(replace(x, "-" => "_")))\\b"
    for element in residues_root["Residue"]
        attributes = attributes_dict(element)
        name = attributes["name"]
        residue_dict[name] = [map(x->get(glossary[name], x, regex(x)),
                              values(attributes_dict(item)))
                              for item in element["Bond"]]
        for alt in alternatives[name]
            residue_dict[alt] = residue_dict[name]
        end
    end
    return residue_dict
end

function pdb_aliases_xml(pdb_bonds)
    regex_set = OrderedSet(regex for list in values(pdb_bonds) for bond in list for regex in bond)

    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Residues")
    regular_expressions = new_child(xroot, "RegularExpressions")
    regex_id = Dict()
    for (id, regex) in enumerate(regex_set)
        regex_entry = new_child(regular_expressions, "Regex")
        set_attributes(regex_entry, Dict("id" => id, "code" => regex))
        regex_id[regex] = id
    end

    for (resname, list) in pdb_bonds
        residue = new_child(xroot, "Residue")
        set_attributes(residue, Dict("name" => resname))
        for bond in list
            ids = Dict("id$i" => regex_id[regex] for (i, regex) in enumerate(bond))
            set_attributes(new_child(residue, "Bond"), ids)
        end
    end
    return xdoc
end

url = "raw.githubusercontent.com/openmm/openmm/master/wrappers/python/simtk/openmm/app/data/"
files = ["residues.xml", "pdbNames.xml"]
pipes = []
for file in files
    push!(pipes, download(url*file))
end

pdb_bonds = pdb_standard_bonds(broadcast(joinpath, @__DIR__, pipes)...)
pdb_aliases = pdb_aliases_xml(pdb_bonds)

io = open(joinpath(@__DIR__, "..", "src", "data", "pdb_aliases.xml"), "w")
write(io, string(pdb_aliases))
close(io)
