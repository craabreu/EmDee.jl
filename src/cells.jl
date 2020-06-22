export Cells, update_cells!

import CUDA
CUDA.allowscalar(false)

mutable struct Cells
    M::Int32
    cutoff::Float32
    action_cells::CUDA.CuArray{Int32,2}
    reaction_cells::CUDA.CuArray{Int32,2}
    head::CUDA.CuArray{Int32,1}
    next::CUDA.CuArray{Int32,1}
    index::CUDA.CuArray{Int32,1}
    population::CUDA.CuArray{Int32,1}
    collected::CUDA.CuArray{Int32,1}
    num_threads::Int32
    num_baskets::Int32
    basket_head::CUDA.CuArray{Int32,1}
    basket_count::CUDA.CuArray{Int32,1}
end

function index2voxel(index, M)
    k, l = divrem(index-1, M*M)
    j, i = divrem(l, M)
    return [i, j, k]
end

function stencil_vectors(rc, action)
    nmax = ceil(Int, rc)
    M = 1+2*nmax
    range = action ? (1:M^3÷2) : (M^3÷2+2:M^3)
    vectors = map(i->index2voxel(i, M) .- nmax, range)
    return filter((x->sum(x.^2) < rc^2) ∘ (x->abs.(x).-1), vectors)
end

cells_per_dimension(L, cutoff, ndiv) = floor(Int32, ndiv*L/cutoff)

function surrounding_cells(L, cutoff, M, action)
    pbc(x) = x < 0 ? x + M : (x >= M ? x - M : x)
    nearby_cell(index, vector) = 1 + (Array{Int32}([1, M, M^2]')*pbc.(index2voxel(index, M) + vector))[1]
    vectors = stencil_vectors(M*cutoff/L, action)
    nearby_cells = [Int32(nearby_cell(i, v)) for v in vectors, i in 1:M^3]
    return nearby_cells
end

function distribute!(head, next, index, population)
    icell = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if icell <= length(head)
        head[icell] = 0
        population[icell] = 0
        for i = 1:length(index)
            if icell == index[i]
                next[i] = head[icell]
                head[icell] = i
                population[icell] += 1
            end
        end
    end
    return nothing
end

function clean_cells!(head, next, index, population, basket_head, basket_count, r, L, M)
    num_threads = CUDA.blockDim().x
    nbytes = sizeof(Int32)*num_threads
    thread_head = CUDA.@cuDynamicSharedMem(Int32, num_threads)
    thread_tail = CUDA.@cuDynamicSharedMem(Int32, num_threads, offset=nbytes)
    thread_count = CUDA.@cuDynamicSharedMem(Int32, num_threads, offset=2*nbytes)
    tid = CUDA.threadIdx().x
    bid = CUDA.blockIdx().x
    thread_head[tid] = 0
    thread_tail[tid] = 0
    thread_count[tid] = 0
    icell = (bid - 1)*num_threads + tid
    if icell <= length(head)
        if head[icell] != 0
            previous = head[icell]
            current = next[previous]
            while current != 0
                s1 = r[1, current]/L
                s2 = r[2, current]/L
                s3 = r[3, current]/L
                v1 = floor(Int32, M*(s1 - floor(s1)))
                v2 = floor(Int32, M*(s2 - floor(s2)))
                v3 = floor(Int32, M*(s3 - floor(s3)))
                index[current] = 1 + v1 + (v2 + v3*M)*M
                if index[current] == icell
                    previous = current
                    current = next[current]
                else
                    removed = current
                    current = next[current]
                    next[previous] = current
                    next[removed] = thread_head[tid]
                    (thread_head[tid] == 0) && (thread_tail[tid] = removed)
                    thread_head[tid] = removed
                    thread_count[tid] += 1
                    population[icell] -= 1
                end
            end
            current = head[icell]
            s1 = r[1, current]/L
            s2 = r[2, current]/L
            s3 = r[3, current]/L
            v1 = floor(Int32, M*(s1 - floor(s1)))
            v2 = floor(Int32, M*(s2 - floor(s2)))
            v3 = floor(Int32, M*(s3 - floor(s3)))
            index[current] = 1 + v1 + (v2 + v3*M)*M
            if index[current] != icell
                removed = current
                head[icell] = next[current]
                next[removed] = thread_head[tid]
                (thread_head[tid] == 0) && (thread_tail[tid] = removed)
                thread_head[tid] = removed
                thread_count[tid] += 1
                population[icell] -= 1
            end
        end
    end
    CUDA.sync_threads()

    s = 1
    while s < num_threads
        if tid % (2*s) == 1
            icell = tid + s
            if thread_head[icell] != 0
                next[thread_tail[icell]] = thread_head[tid]
                (thread_head[tid] == 0) && (thread_tail[tid] = thread_tail[icell])
                thread_head[tid] = thread_head[icell]
            end
            thread_count[tid] += thread_count[icell]
        end
        CUDA.sync_threads()
        s *= 2
    end

    if tid == 1
        basket_head[bid] = thread_head[1]
        basket_count[bid] = thread_count[1]
    end
    return nothing
end

function collect_baskets!(collected, basket_count, basket_head, next)
    ibasket = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if ibasket <= length(basket_head)
        position = 0
        for i = 1:ibasket-1
            position += basket_count[i]
        end
        current = basket_head[ibasket]
        while current != 0
            position += 1
            collected[position] = current
            current = next[current]
        end
        basket_count[ibasket] = position
    end
    return nothing
end

function renew_cells!(head, next, population, collected, count, index)
    icell = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if icell <= length(head)
        for position = 1:last(count)
            i = collected[position]
            if index[i] == icell
                next[i] = head[icell]
                head[icell] = i
                population[icell] += 1
            end
        end
    end
    return nothing
end

function Cells(r::CUDA.CuArray{T,2}, L, cutoff; ndiv=2, num_threads=256) where {T<:Number}
    M = cells_per_dimension(L, cutoff, ndiv)
    action_cells = surrounding_cells(L, cutoff, M, true)
    reaction_cells = surrounding_cells(L, cutoff, M, false)
    s = r'/L
    index = map(x->floor(Int32, x), M*(s .- floor.(s)))*CUDA.cu([1, M, M^2]) .+ 1
    num_cells = M^3
    num_particles = size(r, 2)
    num_baskets = ceil(Int, num_cells/num_threads)
    head = CUDA.zeros(Int32, num_cells)
    next = CUDA.zeros(Int32, num_particles)
    population = CUDA.zeros(Int32, num_cells)
    collected = CUDA.zeros(Int32, num_particles)
    basket_head = CUDA.zeros(Int32, num_baskets)
    basket_count = CUDA.zeros(Int32, num_baskets)
    CUDA.@cuda threads=num_threads blocks=num_baskets distribute!(head, next, index, population)
    return Cells(M, cutoff, action_cells, reaction_cells, head, next, index, population, collected,
                 num_threads, num_baskets, basket_head, basket_count)
end

function update_cells!(cells, r, L)
    CUDA.@cuda(
        threads=cells.num_threads,
        blocks=cells.num_baskets,
        shmem=3*cells.num_threads*sizeof(Int32),
        clean_cells!(
            cells.head, cells.next, cells.index, cells.population,
            cells.basket_head, cells.basket_count,
            r, L, cells.M
        )
    )
    CUDA.@cuda(
        threads=cells.num_threads,
        blocks=ceil(Int, cells.num_baskets/cells.num_threads),
        collect_baskets!(
            cells.collected, cells.basket_count, cells.basket_head, cells.next
        )
    )
    CUDA.@cuda(
        threads=cells.num_threads,
        blocks=cells.num_baskets,
        renew_cells!(
            cells.head, cells.next, cells.population, cells.collected, cells.basket_count, cells.index
        )
    )
    return nothing
end

function find_action_partners1!(action_partner, action_previous, action_count, pair_count,
                               head, next, index, action_cells,
                               rc, s, max_pairs_per_block, max_neighbors_per_atom)
    num_threads = CUDA.blockDim().x
    tid = CUDA.threadIdx().x
    bid = CUDA.blockIdx().x
    overall_previous = CUDA.@cuDynamicSharedMem(Int32, 1)
    tid == 1 && (overall_previous[1] = (bid - 1)*max_pairs_per_block)
    counter = CUDA.@cuDynamicSharedMem(Int32, num_threads, offset=sizeof(Int32))
    num_particles = length(index)
    i = (bid - 1)*num_threads + tid
    if i <= num_particles
        neighbor = CUDA.@cuDynamicSharedMem(Int32, max_neighbors_per_atom, offset=sizeof(Int32)+(num_threads+(tid-1)*max_neighbors_per_atom)*sizeof(Int32))
        s1i = s[1,i]
        s2i = s[2,i]
        s3i = s[3,i]
        rcsq = rc^2
        inside(dx, dy, dz) = (dx-round(dx))^2 + (dy-round(dy))^2 + (dz-round(dz))^2  <= rcsq
        n = 0
        icell = index[i]
        j = head[icell]
        while j != 0
            if j > i && inside(s[1,j]-s1i, s[2,j]-s2i, s[3,j]-s3i)
                n += 1
                if n <= max_neighbors_per_atom
                    neighbor[n] = j
                else
                    # distribute()
                end
            end
            j = next[j]
        end
        for k = 1:size(action_cells, 1)
            jcell = action_cells[k,icell]
            j = head[jcell]
            while j != 0
                if inside(s[1,j]-s1i, s[2,j]-s2i, s[3,j]-s3i)
                    n += 1
                    if n <= max_neighbors_per_atom
                        neighbor[n] = j
                    else
                        # distribute()
                    end
                end
                j = next[j]
            end
        end
        counter[tid] = min(n, max_neighbors_per_atom)
        action_count[i] = n
    end
    CUDA.sync_threads()
    
    if i <= num_particles
        previous_in_block = 0
        for t = 1:tid-1
            previous_in_block += counter[t]
        end
        previous = (bid - 1)*max_pairs_per_block + previous_in_block
        upper_limit = bid*max_pairs_per_block
        for n = 1:counter[tid]
            previous+n <= upper_limit && (action_partner[previous+n] = neighbor[n])
        end
        action_previous[i] = previous
        if tid == num_threads || i == num_particles
            pair_count[bid] = previous_in_block + counter[tid]
        end
    end
    CUDA.sync_threads()
    previous = (bid - 1)*max_pairs_per_block
    for n = pair_count[bid]+tid:num_threads:max_pairs_per_block
        action_partner[previous+n] = 0
    end
    return nothing
end

N = 2000
L = 10.0f0
cutoff = 2.0f0
r = L*CUDA.rand(Float32, 3, N)
cells = Cells(r, L, cutoff)

density = N/L^3
half_sphere_volume = 2π*cells.cutoff^3/3

max_neighbors_per_atom = ceil(Int, density*half_sphere_volume)  # Number of neighbors per particle

# max_neighbors_per_atom*(cells.num_threads + 1)*sizeof(Int32) = 49152
# max_max_neighbors_per_atom = 49152÷(cells.num_threads*sizeof(UInt8)) - 1
max_pairs_per_block = cells.num_threads
num_blocks = ceil(Int32, N/cells.num_threads)
action_partner = CUDA.zeros(Int32, max_pairs_per_block*num_blocks)
action_previous = CUDA.zeros(Int32, N)
action_count = CUDA.zeros(Int32, N)
action_loc = CUDA.zeros(Int32, N)
reaction_previous = CUDA.zeros(Int32, N)
reaction_count = CUDA.zeros(Int32, N)
pair_count = CUDA.zeros(Int32, num_blocks)


# max_neighbors_per_atom = (49152 - cells.num_threads*sizeof(Bool))÷(cells.num_threads*sizeof(Int32)) - 1
CUDA.@cuda(
    threads=cells.num_threads,
    blocks=num_blocks,
    shmem=(1 + max_neighbors_per_atom)*cells.num_threads*sizeof(Int32),
    find_action_partners1!(
        action_partner, action_previous, action_count, pair_count,
        cells.head, cells.next, cells.index, cells.action_cells,
        cutoff/L, r/L, max_pairs_per_block, max_neighbors_per_atom
    )
)
max_neighbors_per_atom
