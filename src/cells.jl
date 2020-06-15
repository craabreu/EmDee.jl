export Cells, update_cells!

using CUDA
CUDA.allowscalar(false)

mutable struct Cells
    M
    neighbors
    head
    next
    index
    collected
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

function neighbor_cells(L, cutoff, M)
    pbc(x) = x < 0 ? x + M : (x >= M ? x - M : x)
    neighbor(index, vector) = (Array{Int32}([1, M, M^2]')*pbc.(index2voxel(index, M) + vector))[1]
    vectors = neighbor_vectors(M*cutoff/L)
    neighbors = [Int32(neighbor(i, v)) for v in vectors, i in 0:(M^3-1)]
    return neighbors
end

function distribute!(head, next, index)
    first = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
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
    nbytes = sizeof(Int32)*blockDim().x
    thread_head = @cuDynamicSharedMem(Int32, blockDim().x)
    thread_tail = @cuDynamicSharedMem(Int32, blockDim().x, offset=nbytes)
    thread_count = @cuDynamicSharedMem(Int32, blockDim().x, offset=2*nbytes)
    tid = threadIdx().x
    bid = blockIdx().x
    num_threads = blockDim().x
    thread_head[tid] = 0
    thread_tail[tid] = 0
    thread_count[tid] = 0
    first = (bid - 1)*num_threads + tid
    stride = num_threads*gridDim().x
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
    first = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
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
    first = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
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

function Cells(r, L, cutoff; ndiv=2, num_threads=256)
    M = cells_per_dimension(L, cutoff, ndiv)
    neighbors = CUDA.cu(neighbor_cells(L, cutoff, M))
    s = r/L
    index = CUDA.cu(Array{Int32}([1, M, M^2]'))*map(x->floor(Int32, x), M*(s .- floor.(s)))
    head = CUDA.fill(Int32(0), M^3)
    next = CUDA.fill(Int32(0), size(r, 2))
    @cuda threads=num_threads blocks=ceil(Int, M^3/num_threads) distribute!(
        head, next, index
    )
    collected = CUDA.fill(Int32(0), size(r, 2))
    return Cells(M, neighbors, head, next, index, collected)
end

function update_cells!(cells, r, L; num_threads=256)
    num_baskets = ceil(Int, cells.M^3/num_threads)
    basket_head = CUDA.fill(Int32(0), num_baskets)
    basket_count = CUDA.fill(Int32(0), num_baskets)
    @cuda threads=num_threads blocks=num_baskets shmem=num_threads*12 clean_cells!(
        cells.head, cells.next, cells.index, basket_head, basket_count, r, L, cells.M
    )
    @cuda threads=num_threads blocks=ceil(Int, num_baskets/num_threads) collect_baskets!(
        cells.collected, basket_count, basket_head, cells.next
    )
    @cuda threads=num_threads blocks=num_baskets renew_cells!(
        cells.head, cells.next, cells.collected, basket_count, cells.index
    )
    return nothing
end
