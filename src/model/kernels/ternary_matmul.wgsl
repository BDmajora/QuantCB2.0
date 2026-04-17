@group(0) @binding(0) var<storage, read> input: array<f32>;        // Activations
@group(0) @binding(1) var<storage, read> weights: array<u32>;      // 2-bit Packed Weights
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // Result

struct Meta {
    d_in: u32,
    d_out: u32,
    batch_size: u32,
};
@group(0) @binding(3) var<uniform> meta: Meta;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x; // Output feature index (d_out)
    let row = id.y; // Batch index

    // Guard against out-of-bounds
    if (row >= meta.batch_size || col >= meta.d_out) { return; }

    var sum: f32 = 0.0;
    let weights_per_u32: u32 = 16;
    let u32_per_row = (meta.d_in + 15) / weights_per_u32;

    // The core loop: iterating through packed u32 chunks
    for (var i: u32 = 0; i < u32_per_row; i = i + 1) {
        let packed_val = weights[col * u32_per_row + i];
        
        for (var bit: u32 = 0; bit < weights_per_u32; bit = bit + 1) {
            let inner_idx = i * weights_per_u32 + bit;
            if (inner_idx >= meta.d_in) { break; }

            // Extract 2 bits for the current weight
            let weight_bits = (packed_val >> (bit * 2u)) & 0x3u;
            let activation = input[row * meta.d_in + inner_idx];

            // FIX: Branchless mapping
            // 01 -> +1.0, 10 -> -1.0, 00/11 -> 0.0
            let w = select(0.0, 1.0, weight_bits == 1u) + select(0.0, -1.0, weight_bits == 2u);
            sum += activation * w;
        }
    }

    // FIX: Apply variance scaling to prevent logit explosion.
    // Scales the output by 1.0 / sqrt(d_in)
    let scale: f32 = 1.0 / sqrt(f32(meta.d_in));
    
    output[row * meta.d_out + col] = sum * scale;
}