import StaticArrays

struct Vec3 <: StaticArrays.FieldVector{3, Float32}
    x::Float32
    y::Float32
    z::Float32
end

@inline shfl_sync(mask, val::Vec3, src) = Vec3(
    CUDA.shfl_sync(mask, val.x, src),
    CUDA.shfl_sync(mask, val.y, src),
    CUDA.shfl_sync(mask, val.z, src)
)
