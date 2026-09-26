"""C ABI kernels for DEAP-compatible variation operators."""

from std.sys import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]


def _mt_twist(state: U32Ptr):
    comptime N = 624
    comptime M = 397
    comptime MATRIX_A = UInt32(0x9908B0DF)
    comptime UPPER_MASK = UInt32(0x80000000)
    comptime LOWER_MASK = UInt32(0x7FFFFFFF)
    for i in range(N - M):
        var word = (state.unsafe_load(i) & UPPER_MASK) | (
            state.unsafe_load(i + 1) & LOWER_MASK
        )
        state.unsafe_store(
            i,
            state.unsafe_load(i + M)
            ^ (word >> UInt32(1))
            ^ (MATRIX_A if word & UInt32(1) else UInt32(0)),
        )
    for i in range(N - M, N - 1):
        var word = (state.unsafe_load(i) & UPPER_MASK) | (
            state.unsafe_load(i + 1) & LOWER_MASK
        )
        state.unsafe_store(
            i,
            state.unsafe_load(i + M - N)
            ^ (word >> UInt32(1))
            ^ (MATRIX_A if word & UInt32(1) else UInt32(0)),
        )
    var word = (state.unsafe_load(N - 1) & UPPER_MASK) | (
        state.unsafe_load(0) & LOWER_MASK
    )
    state.unsafe_store(
        N - 1,
        state.unsafe_load(M - 1)
        ^ (word >> UInt32(1))
        ^ (MATRIX_A if word & UInt32(1) else UInt32(0)),
    )


def _mt_next(state: U32Ptr, mut index: Int) -> UInt32:
    if index >= 624:
        _mt_twist(state)
        index = 0
    var word = state.unsafe_load(index)
    index += 1
    word ^= word >> UInt32(11)
    word ^= (word << UInt32(7)) & UInt32(0x9D2C5680)
    word ^= (word << UInt32(15)) & UInt32(0xEFC60000)
    word ^= word >> UInt32(18)
    return word


def _mt_random(state: U32Ptr, mut index: Int) -> Float64:
    var first = _mt_next(state, index) >> UInt32(5)
    var second = _mt_next(state, index) >> UInt32(6)
    return (
        Float64(first) * 67108864.0 + Float64(second)
    ) * (1.0 / 9007199254740992.0)


@export("mdeap_fill_polynomial_random")
def fill_polynomial_random(
    random_addr: Int,
    state_addr: Int,
    n: Int,
    indpb: Float64,
    start_index: Int,
) abi("C") -> Int:
    var random_values = FPtr(unsafe_from_address=random_addr)
    var state = U32Ptr(unsafe_from_address=state_addr)
    var index = start_index
    for i in range(n):
        if _mt_random(state, index) <= indpb:
            random_values.unsafe_store(i, _mt_random(state, index))
        else:
            random_values.unsafe_store(i, -1.0)
    return index


@export("mdeap_cx_uniform_f64")
def cx_uniform_f64(a_addr: Int, b_addr: Int, mask_addr: Int, n: Int) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var b = FPtr(unsafe_from_address=b_addr)
    var mask = BPtr(unsafe_from_address=mask_addr)
    for i in range(n):
        if mask.unsafe_load(i) != 0:
            var tmp = a.unsafe_load(i)
            a.unsafe_store(i, b.unsafe_load(i))
            b.unsafe_store(i, tmp)


@export("mdeap_cx_uniform_i64")
def cx_uniform_i64(a_addr: Int, b_addr: Int, mask_addr: Int, n: Int) abi("C"):
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var mask = BPtr(unsafe_from_address=mask_addr)
    for i in range(n):
        if mask.unsafe_load(i) != 0:
            var tmp = a.unsafe_load(i)
            a.unsafe_store(i, b.unsafe_load(i))
            b.unsafe_store(i, tmp)


@export("mdeap_cx_uniform_u8")
def cx_uniform_u8(a_addr: Int, b_addr: Int, mask_addr: Int, n: Int) abi("C"):
    var a = BPtr(unsafe_from_address=a_addr)
    var b = BPtr(unsafe_from_address=b_addr)
    var mask = BPtr(unsafe_from_address=mask_addr)
    for i in range(n):
        if mask.unsafe_load(i) != 0:
            var tmp = a.unsafe_load(i)
            a.unsafe_store(i, b.unsafe_load(i))
            b.unsafe_store(i, tmp)


@export("mdeap_cx_pmx")
def cx_pmx(
    a_addr: Int,
    b_addr: Int,
    p1_addr: Int,
    p2_addr: Int,
    n: Int,
    left: Int,
    right: Int,
) abi("C"):
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var p1 = IPtr(unsafe_from_address=p1_addr)
    var p2 = IPtr(unsafe_from_address=p2_addr)
    for i in range(n):
        p1.unsafe_store(Int(a.unsafe_load(i)), Int64(i))
        p2.unsafe_store(Int(b.unsafe_load(i)), Int64(i))
    for i in range(left, right):
        var x1 = a.unsafe_load(i)
        var x2 = b.unsafe_load(i)
        var j1 = Int(p1.unsafe_load(Int(x2)))
        var j2 = Int(p2.unsafe_load(Int(x1)))
        a.unsafe_store(i, x2)
        a.unsafe_store(j1, x1)
        b.unsafe_store(i, x1)
        b.unsafe_store(j2, x2)
        p1.unsafe_store(Int(x1), Int64(j1))
        p1.unsafe_store(Int(x2), Int64(i))
        p2.unsafe_store(Int(x1), Int64(i))
        p2.unsafe_store(Int(x2), Int64(j2))


@export("mdeap_cx_upmx")
def cx_upmx(
    a_addr: Int,
    b_addr: Int,
    p1_addr: Int,
    p2_addr: Int,
    mask_addr: Int,
    n: Int,
) abi("C"):
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var p1 = IPtr(unsafe_from_address=p1_addr)
    var p2 = IPtr(unsafe_from_address=p2_addr)
    var mask = BPtr(unsafe_from_address=mask_addr)
    for i in range(n):
        p1.unsafe_store(Int(a.unsafe_load(i)), Int64(i))
        p2.unsafe_store(Int(b.unsafe_load(i)), Int64(i))
    for i in range(n):
        if mask.unsafe_load(i) != 0:
            var x1 = a.unsafe_load(i)
            var x2 = b.unsafe_load(i)
            var j1 = Int(p1.unsafe_load(Int(x2)))
            var j2 = Int(p2.unsafe_load(Int(x1)))
            a.unsafe_store(i, x2)
            a.unsafe_store(j1, x1)
            b.unsafe_store(i, x1)
            b.unsafe_store(j2, x2)
            p1.unsafe_store(Int(x1), Int64(j1))
            p1.unsafe_store(Int(x2), Int64(i))
            p2.unsafe_store(Int(x1), Int64(i))
            p2.unsafe_store(Int(x2), Int64(j2))


@export("mdeap_cx_ordered")
def cx_ordered(
    a_addr: Int,
    b_addr: Int,
    holes1_addr: Int,
    holes2_addr: Int,
    n: Int,
    left: Int,
    right: Int,
) abi("C"):
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var holes1 = BPtr(unsafe_from_address=holes1_addr)
    var holes2 = BPtr(unsafe_from_address=holes2_addr)
    for i in range(n):
        holes1.unsafe_store(i, 1)
        holes2.unsafe_store(i, 1)
    for i in range(n):
        if i < left or i > right:
            holes1.unsafe_store(Int(b.unsafe_load(i)), 0)
            holes2.unsafe_store(Int(a.unsafe_load(i)), 0)
    var k1 = right + 1
    var k2 = right + 1
    for i in range(n):
        var source = (i + right + 1) % n
        var x1 = a.unsafe_load(source)
        var x2 = b.unsafe_load(source)
        if holes1.unsafe_load(Int(x1)) == 0:
            a.unsafe_store(k1 % n, x1)
            k1 += 1
        if holes2.unsafe_load(Int(x2)) == 0:
            b.unsafe_store(k2 % n, x2)
            k2 += 1
    for i in range(left, right + 1):
        var tmp = a.unsafe_load(i)
        a.unsafe_store(i, b.unsafe_load(i))
        b.unsafe_store(i, tmp)


@export("mdeap_cx_blend")
def cx_blend(
    a_addr: Int,
    b_addr: Int,
    random_addr: Int,
    n: Int,
    alpha: Float64,
) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var b = FPtr(unsafe_from_address=b_addr)
    var random_values = FPtr(unsafe_from_address=random_addr)
    for i in range(n):
        var x1 = a.unsafe_load(i)
        var x2 = b.unsafe_load(i)
        var gamma = (1.0 + 2.0 * alpha) * random_values.unsafe_load(i) - alpha
        a.unsafe_store(i, (1.0 - gamma) * x1 + gamma * x2)
        b.unsafe_store(i, gamma * x1 + (1.0 - gamma) * x2)


@export("mdeap_cx_sbx")
def cx_sbx(
    a_addr: Int,
    b_addr: Int,
    random_addr: Int,
    n: Int,
    eta: Float64,
) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var b = FPtr(unsafe_from_address=b_addr)
    var random_values = FPtr(unsafe_from_address=random_addr)
    var power = 1.0 / (eta + 1.0)
    for i in range(n):
        var rand = random_values.unsafe_load(i)
        var beta = 2.0 * rand
        if rand > 0.5:
            beta = 1.0 / (2.0 * (1.0 - rand))
        beta = beta**power
        var x1 = a.unsafe_load(i)
        var x2 = b.unsafe_load(i)
        a.unsafe_store(i, 0.5 * (((1.0 + beta) * x1) + ((1.0 - beta) * x2)))
        b.unsafe_store(i, 0.5 * (((1.0 - beta) * x1) + ((1.0 + beta) * x2)))


@export("mdeap_cx_sbx_bounded")
def cx_sbx_bounded(
    a_addr: Int,
    b_addr: Int,
    low_addr: Int,
    up_addr: Int,
    random_addr: Int,
    active_addr: Int,
    swap_addr: Int,
    n: Int,
    eta: Float64,
) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var b = FPtr(unsafe_from_address=b_addr)
    var low = FPtr(unsafe_from_address=low_addr)
    var up = FPtr(unsafe_from_address=up_addr)
    var random_values = FPtr(unsafe_from_address=random_addr)
    var active = BPtr(unsafe_from_address=active_addr)
    var swaps = BPtr(unsafe_from_address=swap_addr)
    var power = 1.0 / (eta + 1.0)
    for i in range(n):
        if active.unsafe_load(i) == 0:
            continue
        var first = min(a.unsafe_load(i), b.unsafe_load(i))
        var second = max(a.unsafe_load(i), b.unsafe_load(i))
        var width = second - first
        var rand = random_values.unsafe_load(i)
        var beta = 1.0 + (2.0 * (first - low.unsafe_load(i)) / width)
        var coeff = 2.0 - beta ** (-(eta + 1.0))
        var beta_q = (rand * coeff) ** power
        if rand > 1.0 / coeff:
            beta_q = (1.0 / (2.0 - rand * coeff)) ** power
        var c1 = 0.5 * (first + second - beta_q * width)
        beta = 1.0 + (2.0 * (up.unsafe_load(i) - second) / width)
        coeff = 2.0 - beta ** (-(eta + 1.0))
        beta_q = (rand * coeff) ** power
        if rand > 1.0 / coeff:
            beta_q = (1.0 / (2.0 - rand * coeff)) ** power
        var c2 = 0.5 * (first + second + beta_q * width)
        c1 = min(max(c1, low.unsafe_load(i)), up.unsafe_load(i))
        c2 = min(max(c2, low.unsafe_load(i)), up.unsafe_load(i))
        if swaps.unsafe_load(i) != 0:
            a.unsafe_store(i, c2)
            b.unsafe_store(i, c1)
        else:
            a.unsafe_store(i, c1)
            b.unsafe_store(i, c2)


@export("mdeap_mut_gaussian")
def mut_gaussian(
    values_addr: Int,
    delta_addr: Int,
    active_addr: Int,
    n: Int,
) abi("C"):
    var values = FPtr(unsafe_from_address=values_addr)
    var delta = FPtr(unsafe_from_address=delta_addr)
    var active = BPtr(unsafe_from_address=active_addr)
    for i in range(n):
        if active.unsafe_load(i) != 0:
            values.unsafe_store(
                i, values.unsafe_load(i) + delta.unsafe_load(i)
            )


def _polynomial_vector[W: Int](
    x: SIMD[DType.float64, W],
    lower: SIMD[DType.float64, W],
    upper: SIMD[DType.float64, W],
    rand: SIMD[DType.float64, W],
    eta: Float64,
) -> SIMD[DType.float64, W]:
    var width = upper - lower
    var delta1 = (x - lower) / width
    var delta2 = (upper - x) / width
    var eta_vector = SIMD[DType.float64, W](eta + 1.0)
    var mutation_power = SIMD[DType.float64, W](1.0 / (eta + 1.0))
    var left_xy = 1.0 - delta1
    var left_val = (
        2.0 * rand + (1.0 - 2.0 * rand) * left_xy**eta_vector
    )
    var left_delta = left_val**mutation_power - 1.0
    var right_xy = 1.0 - delta2
    var right_val = (
        2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * right_xy**eta_vector
    )
    var right_delta = 1.0 - right_val**mutation_power
    var delta_q = rand.lt(0.5).select(left_delta, right_delta)
    return min(max(x + delta_q * width, lower), upper)


def _polynomial_scalar(
    x: Float64,
    lower: Float64,
    upper: Float64,
    rand: Float64,
    eta: Float64,
) -> Float64:
    var width = upper - lower
    var mutation_power = 1.0 / (eta + 1.0)
    var delta_q: Float64
    if rand < 0.5:
        var xy = 1.0 - (x - lower) / width
        var val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
        delta_q = val**mutation_power - 1.0
    else:
        var xy = 1.0 - (upper - x) / width
        var val = (
            2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
        )
        delta_q = 1.0 - val**mutation_power
    return min(max(x + delta_q * width, lower), upper)


@export("mdeap_mut_polynomial")
def mut_polynomial(
    values_addr: Int,
    low_addr: Int,
    up_addr: Int,
    random_addr: Int,
    indices_addr: Int,
    active_count: Int,
    eta: Float64,
) abi("C"):
    var values = FPtr(unsafe_from_address=values_addr)
    var low = FPtr(unsafe_from_address=low_addr)
    var up = FPtr(unsafe_from_address=up_addr)
    var random_values = FPtr(unsafe_from_address=random_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    comptime W = simd_width_of[DType.float64]()
    var vector_end = (active_count // W) * W
    for i in range(0, vector_end, W):
        var gene_indices = indices.unsafe_load[width=W](i)
        var x = values.unsafe_gather(gene_indices)
        var lower = low.unsafe_gather(gene_indices)
        var upper = up.unsafe_gather(gene_indices)
        var rand = random_values.unsafe_gather(gene_indices)
        values.unsafe_scatter(
            gene_indices, _polynomial_vector(x, lower, upper, rand, eta)
        )
    for i in range(vector_end, active_count):
        var gene = Int(indices.unsafe_load(i))
        values.unsafe_store(
            gene,
            _polynomial_scalar(
                values.unsafe_load(gene),
                low.unsafe_load(gene),
                up.unsafe_load(gene),
                random_values.unsafe_load(gene),
                eta,
            ),
        )


@export("mdeap_mut_polynomial_scalar_bounds")
def mut_polynomial_scalar_bounds(
    values_addr: Int,
    random_addr: Int,
    indices_addr: Int,
    active_count: Int,
    eta: Float64,
    lower: Float64,
    upper: Float64,
) abi("C"):
    var values = FPtr(unsafe_from_address=values_addr)
    var random_values = FPtr(unsafe_from_address=random_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    comptime W = simd_width_of[DType.float64]()
    var vector_end = (active_count // W) * W
    var lower_vector = SIMD[DType.float64, W](lower)
    var upper_vector = SIMD[DType.float64, W](upper)
    for i in range(0, vector_end, W):
        var gene_indices = indices.unsafe_load[width=W](i)
        var x = values.unsafe_gather(gene_indices)
        var rand = random_values.unsafe_gather(gene_indices)
        values.unsafe_scatter(
            gene_indices,
            _polynomial_vector(x, lower_vector, upper_vector, rand, eta),
        )
    for i in range(vector_end, active_count):
        var gene = Int(indices.unsafe_load(i))
        values.unsafe_store(
            gene,
            _polynomial_scalar(
                values.unsafe_load(gene),
                lower,
                upper,
                random_values.unsafe_load(gene),
                eta,
            ),
        )


@export("mdeap_mut_polynomial_scalar_bounds_gpu")
def mut_polynomial_scalar_bounds_gpu(
    values_addr: Int,
    random_addr: Int,
    indices_addr: Int,
    n: Int,
    active_count: Int,
    eta: Float64,
    lower: Float64,
    upper: Float64,
) abi("C") -> Int:
    # This toolchain ships no GPU host API (no DeviceContext, no enqueue_*), so
    # the `device="gpu"` entry point runs the same serial kernel on the CPU.
    # See README: the port is CPU-only by necessity, not by oversight.
    mut_polynomial_scalar_bounds(
        values_addr, random_addr, indices_addr, active_count, eta, lower, upper
    )
    return 1


@export("mdeap_mut_shuffle")
def mut_shuffle(values_addr: Int, targets_addr: Int, n: Int) abi("C"):
    var values = IPtr(unsafe_from_address=values_addr)
    var targets = IPtr(unsafe_from_address=targets_addr)
    for i in range(n):
        var target = Int(targets.unsafe_load(i))
        if target >= 0:
            var tmp = values.unsafe_load(i)
            values.unsafe_store(i, values.unsafe_load(target))
            values.unsafe_store(target, tmp)


@export("mdeap_mut_flip")
def mut_flip(values_addr: Int, active_addr: Int, n: Int) abi("C"):
    var values = BPtr(unsafe_from_address=values_addr)
    var active = BPtr(unsafe_from_address=active_addr)
    for i in range(n):
        if active.unsafe_load(i) != 0:
            var flipped: UInt8 = 0 if values.unsafe_load(i) != 0 else 1
            values.unsafe_store(i, flipped)


@export("mdeap_mut_uniform_int")
def mut_uniform_int(
    values_addr: Int,
    replacements_addr: Int,
    active_addr: Int,
    n: Int,
) abi("C"):
    var values = IPtr(unsafe_from_address=values_addr)
    var replacements = IPtr(unsafe_from_address=replacements_addr)
    var active = BPtr(unsafe_from_address=active_addr)
    for i in range(n):
        if active.unsafe_load(i) != 0:
            values.unsafe_store(i, replacements.unsafe_load(i))


@export("mdeap_mut_es")
def mut_es(
    values_addr: Int,
    strategy_addr: Int,
    factors_addr: Int,
    normals_addr: Int,
    active_addr: Int,
    n: Int,
) abi("C"):
    var values = FPtr(unsafe_from_address=values_addr)
    var strategy = FPtr(unsafe_from_address=strategy_addr)
    var factors = FPtr(unsafe_from_address=factors_addr)
    var normals = FPtr(unsafe_from_address=normals_addr)
    var active = BPtr(unsafe_from_address=active_addr)
    for i in range(n):
        if active.unsafe_load(i) != 0:
            strategy.unsafe_store(
                i, strategy.unsafe_load(i) * factors.unsafe_load(i)
            )
            values.unsafe_store(
                i,
                values.unsafe_load(i)
                + strategy.unsafe_load(i) * normals.unsafe_load(i),
            )
