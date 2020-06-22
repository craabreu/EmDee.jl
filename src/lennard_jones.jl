struct LennardJonesAtom
    half_σ::Float32
    sqrt_4ε::Float32
    LennardJonesAtom(ε, σ) = new(0.5σ, sqrt(4ε))
end

struct LennardJonesModel
    rc²::Float32
    rs²::Float32
    δ⁻²::Float32
    LennardJonesModel(rc, rs) = new(rc^2, rs^2, 1/(rc^2 - rs^2))
end

@inline function interaction(r²::Float32,
                             model::LennardJonesModel,
                             atom_i::LennardJonesAtom,
                             atom_j::LennardJonesAtom)::Tuple{Float32,Float32}
    σ = atom_i.half_σ + atom_j.half_σ
    ε4 = atom_i.sqrt_4ε*atom_j.sqrt_4ε
    s⁻² = σ*σ/r²
    s⁻⁶ = s⁻²*s⁻²*s⁻²
    s⁻¹² = s⁻⁶*s⁻⁶
    E = ε4*(s⁻¹² - s⁻⁶)
    r⁻¹E′ = -6ε4*(2s⁻¹² - s⁻⁶)/r²
    x = (r² - model.rs²)*model.δ⁻²
    x *= (sign(x) - sign(x-1))/2
    x² = x*x
    g = 1 + x*x²*(15x - 6x² - 10)
    r⁻¹g′ = -60x²*(2x - x² - 1)*model.δ⁻²
    return E*g, E*r⁻¹g′ + r⁻¹E′*g
end
