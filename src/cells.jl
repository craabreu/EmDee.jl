export Cells, update_cells!

import CUDA
CUDA.allowscalar(false)

mutable struct Cells
    M::Int32
    cutoff::Float32
    nearby_cells::CUDA.CuArray{Int32,2}
    head::CUDA.CuArray{Int32,1}
    next::CUDA.CuArray{Int32,1}
    index::CUDA.CuArray{Int32,1}
    collected::CUDA.CuArray{Int32,1}
    num_threads::Int32
    num_baskets::Int32
    basket_head::CUDA.CuArray{Int32,1}
    basket_count::CUDA.CuArray{Int32,1}
end

function index2voxel(index, M)
    k, l = divrem(index, M*M)
    j, i = divrem(l, M)
    return [i, j, k]
end

function stencil_vectors(rc)
    nmax = ceil(Int, rc)
    M = 1+2*nmax
    vectors = map(i->index2voxel(i, M) .- nmax, (M^3÷2+1):(M^3-1))
    return filter((x->sum(x.^2) < rc^2) ∘ (x->abs.(x).-1), vectors)
end

cells_per_dimension(L, cutoff, ndiv) = floor(Int32, ndiv*L/cutoff)

function surrounding_cells(L, cutoff, M)
    pbc(x) = x < 0 ? x + M : (x >= M ? x - M : x)
    nearby_cell(index, vector) = (Array{Int32}([1, M, M^2]')*pbc.(index2voxel(index, M) + vector))[1]
    vectors = stencil_vectors(M*cutoff/L)
    nearby_cells = [Int32(nearby_cell(i, v)) for v in vectors, i in 0:(M^3-1)]
    return nearby_cells
end

function distribute!(head, next, index)
    first = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for icell = first:stride:length(head)
        head[icell] = 0
        for i = 1:length(index)
            if icell == index[i] + 1
                next[i] = head[icell]
                head[icell] = i
            end
        end
    end
    return nothing
end

function clean_cells!(head, next, index, basket_head, basket_count, r, L, M)
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
    first = (bid - 1)*num_threads + tid
    stride = num_threads*CUDA.gridDim().x
    for icell = first:stride:length(head)
        if head[icell] != 0
            cell_index = icell - 1
            previous = head[icell]
            current = next[previous]
            while current != 0
                s1 = r[1, current]/L
                s2 = r[2, current]/L
                s3 = r[3, current]/L
                v1 = floor(Int32, M*(s1 - floor(s1)))
                v2 = floor(Int32, M*(s2 - floor(s2)))
                v3 = floor(Int32, M*(s3 - floor(s3)))
                index[current] = v1 + (v2 + v3*M)*M
                if index[current] == cell_index
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
                end
            end
            current = head[icell]
            s1 = r[1, current]/L
            s2 = r[2, current]/L
            s3 = r[3, current]/L
            v1 = floor(Int32, M*(s1 - floor(s1)))
            v2 = floor(Int32, M*(s2 - floor(s2)))
            v3 = floor(Int32, M*(s3 - floor(s3)))
            index[current] = v1 + (v2 + v3*M)*M
            if index[current] != cell_index
                removed = current
                head[icell] = next[current]
                next[removed] = thread_head[tid]
                (thread_head[tid] == 0) && (thread_tail[tid] = removed)
                thread_head[tid] = removed
                thread_count[tid] += 1
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

function collect_baskets!(dislocated, basket_count, basket_head, next)
    first = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for ibasket = first:stride:length(basket_head)
        position = 0
        for i = 1:ibasket-1
            position += basket_count[i]
        end
        current = basket_head[ibasket]
        while current != 0
            position += 1
            dislocated[position] = current
            current = next[current]
        end
        basket_count[ibasket] = position
    end
    return nothing
end

function renew_cells!(head, next, collected, count, index)
    first = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for icell = first:stride:length(head)
        cell_index = icell - 1
        for position = 1:last(count)
            i = collected[position]
            if index[i] == cell_index
                next[i] = head[icell]
                head[icell] = i
            end
        end
    end
    return nothing
end

function Cells(r::CUDA.CuArray{T,2}, L, cutoff; ndiv=2, num_threads=256) where {T<:Number}
    M = cells_per_dimension(L, cutoff, ndiv)
    nearby_cells = surrounding_cells(L, cutoff, M)
    s = r'/L
    index = map(x->floor(Int32, x), M*(s .- floor.(s)))*CUDA.cu([1, M, M^2])
    num_cells = M^3
    num_particles = size(r, 2)
    num_baskets = ceil(Int, num_cells/num_threads)
    head = CUDA.zeros(Int32, num_cells)
    next = CUDA.zeros(Int32, num_particles)
    collected = CUDA.zeros(Int32, num_particles)
    basket_head = CUDA.zeros(Int32, num_baskets)
    basket_count = CUDA.zeros(Int32, num_baskets)
    CUDA.@cuda threads=num_threads blocks=num_baskets distribute!(head, next, index)
    return Cells(M, cutoff, nearby_cells, head, next, index, collected,
                 num_threads, num_baskets, basket_head, basket_count)
end

function update_cells!(cells, r, L)
    CUDA.@cuda threads=cells.num_threads blocks=cells.num_baskets shmem=12*cells.num_threads clean_cells!(
        cells.head, cells.next, cells.index, cells.basket_head, cells.basket_count, r, L, cells.M
    )
    CUDA.@cuda threads=cells.num_threads blocks=ceil(Int, cells.num_baskets/cells.num_threads) collect_baskets!(
        cells.collected, cells.basket_count, cells.basket_head, cells.next
    )
    CUDA.@cuda threads=cells.num_threads blocks=cells.num_baskets renew_cells!(
        cells.head, cells.next, cells.collected, cells.basket_count, cells.index
    )
    return nothing
end
