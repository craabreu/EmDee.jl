using nauty_jll
using Parameters

const LIB_FILE = libnautyW1
const WORDSIZE = 64

@with_kw mutable struct OptionBlk
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
    dispatch::Ptr{Cvoid}=cglobal((:dispatch_graph, LIB_FILE), Nothing)
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

const OPTION_BLK = OptionBlk(getcanon=true)
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
    ccall(
        (:densenauty, LIB_FILE),
        Cvoid,
        (Ptr{UInt64}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ref{OptionBlk}, Ref{StatsBlk}, Cint, Cint, Ptr{UInt64}),
         g.chunks, lab, ptn, orbits, OPTION_BLK, STATS_BLK, m, n, cg.chunks
    )
    return lab .+ 1, graph2matrix(cg)
end
