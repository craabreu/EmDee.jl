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

function matrix2graph(matrix)
    n = size(matrix, 1)
    nw = WORDSIZE*cld(n, WORDSIZE)
    expanded = vcat(matrix, falses(nw-n, n))
    expanded.chunks .= expanded[end:-1:1, end:-1:1].chunks[end:-1:1]
    return expanded
end

function graph2matrix(graph)
    n = size(graph, 2)
    nw = WORDSIZE*cld(n, WORDSIZE)
    reverted = BitArray{2}(undef, size(graph)...)
    reverted.chunks = graph.chunks[end:-1:1]
    return reverted[end:-1:end+1-n, end:-1:end+1-n]
end

function canonical_form(matrix, colors; atol=0.1)
    n = size(matrix, 1)
    m = cld(n, WORDSIZE)
    lab = Vector{Cint}(sortperm(colors))
    ptn = Vector{Cint}(abs.(diff(colors[lab])) .<= atol)
    lab .= lab .- 1
    push!(ptn, 0)
    orbits = Vector{Cint}(undef, n)
    g = matrix2graph(matrix)
    cg = similar(g)
    dispatch_graph = cglobal((:dispatch_graph, LIB_FILE), Nothing)
    option_block = OptionBlk(getcanon=true, defaultptn=false, dispatch=dispatch_graph)
    ccall(
        (:densenauty, LIB_FILE),
        Cvoid,
        (Ptr{UInt64}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ref{OptionBlk}, Ref{StatsBlk}, Cint, Cint, Ptr{UInt64}),
        g.chunks, lab, ptn, orbits, option_block, StatsBlk(), m, n, cg.chunks
    )
    return lab .+ 1, graph2matrix(cg)
end