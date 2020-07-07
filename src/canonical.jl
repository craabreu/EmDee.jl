using nauty_jll
using Parameters

const LIB_FILE = libnautyL0
const WORDSIZE = 64

@with_kw struct OptionBlk
    getcanon::Cint=false
    digraph::Cint=false
    writeautoms::Cint=false
    writemarkers::Cint=false
    defaultptn::Cint=true
    cartesian::Cint=false
    linelength::Cint=78
    outfile::Ptr{Cvoid}=C_NULL
    userrefproc::Ptr{Cvoid}=C_NULL
    userautomproc::Ptr{Cvoid}=C_NULL
    userlevelproc::Ptr{Cvoid}=C_NULL
    usernodeproc::Ptr{Cvoid}=C_NULL
    usercanonproc::Ptr{Cvoid}=C_NULL
    invarproc::Ptr{Cvoid}=C_NULL
    tc_level::Cint=100
    mininvarlevel::Cint=0
    maxinvarlevel::Cint=1
    invararg::Cint=0
    dispatch::Ptr{Cvoid}=C_NULL
    schreier::Cint=false
    extra_options::Ptr{Cvoid}=C_NULL
end

@with_kw struct StatsBlk
    grpsize1::Cdouble=0
    grpsize2::Cint=0
    numorbits::Cint=0
    numgenerators::Cint=0
    errstatus::Cint=0
    numnodes::Culong=0
    numbadleaves::Culong=0
    maxlevel::Cint=0
    tctotal::Culong=0
    canupdates::Culong=0
    invapplics::Culong=0
    invsuccesses::Culong=0
    invarsuclevel::Cint=0
end

const STATS_BLK = StatsBlk()

function matrix2graph(matrix)
    n = size(matrix, 1)
    nw = WORDSIZE*cld(n, WORDSIZE)
    return vcat(falses(nw-n, n), matrix[n:-1:1,:])
end

function graph2matrix(graph)
    n = size(graph, 2)
    nw = WORDSIZE*cld(n, WORDSIZE)
    return graph[nw:-1:nw-n+1,:]
end

function canonical_form(matrix)
    n = size(matrix, 1)
    m = cld(n, WORDSIZE)
    lab = Vector{Cint}(undef, n)
    ptn = Vector{Cint}(undef, n)
    orbits = Vector{Cint}(undef, n)
    g = matrix2graph(matrix)
    cg = similar(g)
    dispatch_graph = cglobal((:dispatch_graph, LIB_FILE), Nothing)
    option_block = OptionBlk(getcanon=true, dispatch=dispatch_graph)
    ccall(
        (:densenauty, LIB_FILE),
        Cvoid,
        (Ptr{UInt64}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ref{OptionBlk}, Ref{StatsBlk}, Cint, Cint, Ptr{UInt64}),
        g.chunks, lab, ptn, orbits, option_block, STATS_BLK, m, n, cg.chunks
    )
    return reverse(lab .+ 1), graph2matrix(cg)[n:-1:1, n:-1:1]
end

function canonical_mapping!(residue_atoms, bonds)
    num_atoms = sum(length.(residue_atoms))
    atom_residue = Vector{UInt128}(undef, num_atoms)
    internal_map = Vector{UInt128}(undef, num_atoms)
    for (index, atom_indices) in enumerate(residue_atoms)
        atom_residue[atom_indices] .= index
        internal_map[atom_indices] .= collect(1:length(atom_indices))
    end
    adjacency = map(n->falses(n, n), map(length, residue_atoms))
    for (atom_1, atom_2) in eachcol(bonds)
        index = atom_residue[atom_1]
        if atom_residue[atom_2] == index
            i = internal_map[atom_1]
            j = internal_map[atom_2]
            adjacency[index][i, j] = adjacency[index][j, i] = true
        end
    end
    for index = 1:length(residue_atoms)
        canonical_order, canonical_adjacency = canonical_form(adjacency[index])
        residue_atoms[index] .= residue_atoms[index][canonical_order]
        adjacency[index] .= canonical_adjacency
    end
    mapping = invperm(vcat(residue_atoms...)) .- 1
    return adjacency, mapping
end
