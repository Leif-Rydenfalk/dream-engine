// --- STRUCTS ---

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Neuron {
    // 128-bit Semantic Hypervector (SDR Identity)
    semantic: vec4<u32>,  // 16 bytes
    
    // Spatial Position (-1.0 to 1.0)
    pos: vec2<f32>,       // 8 bytes
    
    // State
    voltage: f32,         // 4 bytes (Activation/Error)
    prediction: f32,      // 4 bytes (Memoized State/Visualization)
    precision_value: f32, // 4 bytes (Plasticity factor)
    layer: u32,           // 4 bytes (0..6)
    
    // Padding
    pad0: f32,            
    pad1: f32,            
};

struct SimParams {
    neuron_count: u32,
    time: f32,
    dt: f32,
    grid_dim: u32,
    train_mode: u32, 
    use_camera: u32, 
    pad0: f32,
    pad1: f32,
};

struct LineVertex {
    pos: vec4<f32>,
    color: vec4<f32>,
};

// --- BINDINGS ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read_write> spatial_grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> line_buffer: array<LineVertex>;

@group(0) @binding(4) var input_tex: texture_2d<f32>;
@group(0) @binding(5) var input_sampler: sampler;
@group(0) @binding(6) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var prediction_tex: texture_storage_2d<rgba32float, write>; // L6 writes here
@group(0) @binding(8) var prediction_tex_read: texture_2d<f32>;                // L0 reads here

// --- UTILITY ---

fn hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(hash(seed)) / 4294967295.0;
}

// Hamming similarity: 1.0 = Identical, 0.0 = Opposite
fn hamming_similarity(a: vec4<u32>, b: vec4<u32>) -> f32 {
    let diff = a ^ b;
    let bits = countOneBits(diff.x) + countOneBits(diff.y) + countOneBits(diff.z) + countOneBits(diff.w);
    return max(0.0, 1.0 - f32(bits) / 64.0); 
}

fn hash_to_grid(pos: vec2<f32>) -> u32 {
    let grid_size = f32(params.grid_dim);
    let u = (pos.x + 1.0) * 0.5;
    let v = (pos.y + 1.0) * 0.5;
    let x = u32(clamp(u * grid_size, 0.0, grid_size - 1.0));
    let y = u32(clamp(v * grid_size, 0.0, grid_size - 1.0));
    return x + y * params.grid_dim;
}

// --- COMPUTE SHADERS ---

@compute @workgroup_size(64)
fn cs_init_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    var n: Neuron;
    let seed = idx * 7123u;
    
    // Random SDR Semantic
    n.semantic = vec4<u32>(hash(seed), hash(seed+1u), hash(seed+2u), hash(seed+3u));
    n.voltage = 0.0;
    n.prediction = rand_f32(seed+6u) * 0.5 + 0.25;
    
    // --- LAYER ARCHITECTURE ---
    // L0: Retina Input (25,600 neurons, dense grid)
    // L1-L3: Encoder Path (Compression)
    // L3: Bottleneck
    // L4-L5: Decoder Path (Reconstruction)
    // L6: Prediction Output (25,600 neurons, dense grid)
    
    let retina_count = 25600u;
    
    if (idx < retina_count) {
        // L0: Input Retina
        n.layer = 0u; 
        let dim = 160u;
        let rx = idx % dim;
        let ry = idx / dim;
        let u = f32(rx) / f32(dim);
        let v = f32(ry) / f32(dim);
        n.pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.precision_value = 10.0; 
    } else if (idx < retina_count * 2u) {
        // L6: Output Prediction
        n.layer = 6u; 
        let i = idx - retina_count;
        let dim = 160u;
        let rx = i % dim;
        let ry = i / dim;
        let u = f32(rx) / f32(dim);
        let v = f32(ry) / f32(dim);
        n.pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.precision_value = 10.0;
    } else {
        // L1..L5: Hidden Layers (Bottleneck)
        let rem_idx = idx - retina_count * 2u;
        let rem_total = params.neuron_count - retina_count * 2u;
        
        // Distribute evenly across 5 hidden layers
        let split = rem_total / 5u;
        let l_offset = rem_idx / split;
        n.layer = clamp(l_offset + 1u, 1u, 5u);
        
        // Topological Map + Jitter
        // We seed position based on index to keep spatial coherence, but add randomness
        n.pos = vec2<f32>(rand_f32(seed+4u), rand_f32(seed+5u)) * 2.0 - 1.0;
        n.precision_value = 1.0;
    }
    
    neurons[idx] = n;
}

@compute @workgroup_size(64)
fn cs_clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    // The array is now 8x larger, so we must ensure we cover it all.
    // Note: Dispatch size in Rust is based on GRID*GRID.
    // We need to clear GRID*GRID*8.
    // Quick hack: Loop inside the shader or just trust the buffer size check?
    // Better: In Rust, increase dispatch size x8 OR loop here.
    // Let's Loop here to avoid changing Rust dispatch logic too much.
    
    let layer_stride = params.grid_dim * params.grid_dim;
    if (idx >= layer_stride) { return; }

    for (var i = 0u; i < 8u; i++) {
        atomicStore(&spatial_grid[idx + i * layer_stride], 0xFFFFFFFFu);
    }
}

@compute @workgroup_size(64)
fn cs_populate_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    let n = neurons[idx];
    let layer_stride = params.grid_dim * params.grid_dim;
    
    // Offset by neuron layer. Safety clamp to 7.
    let layer_offset = clamp(n.layer, 0u, 7u) * layer_stride;
    let grid_idx = hash_to_grid(n.pos);
    
    atomicStore(&spatial_grid[grid_idx + layer_offset], idx);
}

@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    var n = neurons[idx];
    
    // --- L0: RETINA (COMPARE REALITY vs PREDICTION) ---
    if (n.layer == 0u) {
        let uv = (n.pos + 1.0) * 0.5;
        
        // 1. Sample Reality
        var reality = 0.0;
        if (params.use_camera == 1u) {
            reality = textureSampleLevel(input_tex, input_sampler, uv, 0.0).r;
        } else {
            // Dream noise if camera off
            let t = params.time;
            reality = (rand_f32(idx + u32(t * 60.0)) - 0.5) * 0.1 + 0.5;
        }
        
        // 2. Sample Prediction (Top-Down from L6)
        let prediction = textureLoad(prediction_tex_read, vec2<i32>(uv * 512.0), 0).r;
        
        // 3. Compute Error
        // L0 Voltage = Prediction Error. This drives L1.
        n.voltage = reality - prediction; 
        
        // Store reality in prediction slot just for visualization debugging
        n.prediction = reality; 
        
        neurons[idx] = n;
        return;
    }

    // --- L1..L6: HIERARCHICAL CHAIN ---
    // Each layer samples from the layer conceptually "below" it in the feedforward chain.
    // Chain: L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
    // L1-L3: Encoder (Abstracting Error)
    // L4-L6: Decoder (Reconstructing Prediction)

    let source_layer = n.layer - 1u;
    let layer_stride = params.grid_dim * params.grid_dim;
    let source_grid_offset = source_layer * layer_stride;
    
    var accum = 0.0;
    var weight_sum = 0.0;
    var best_source_idx = 0xFFFFFFFFu;
    var max_sim = 0.0;
    
    let grid_dim = f32(params.grid_dim);
    let center_u = (n.pos.x + 1.0) * 0.5 * grid_dim;
    let center_v = (n.pos.y + 1.0) * 0.5 * grid_dim;
    
    // Receptive Field Search (Spatial Locality)
    let rf_rad = 2;
    for (var dy = -rf_rad; dy <= rf_rad; dy++) {
        for (var dx = -rf_rad; dx <= rf_rad; dx++) {
            let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
            let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
            let g_idx = gx + gy * params.grid_dim + source_grid_offset; 
            let src_idx = atomicLoad(&spatial_grid[g_idx]);
            if (src_idx != 0xFFFFFFFFu) {
                let src = neurons[src_idx];
                if (src.layer == source_layer) {
                    let dist = distance(n.pos, src.pos);
                    if (dist < 0.15) { // RF Radius
                        // Filter by Semantic Similarity
                        let sim = hamming_similarity(n.semantic, src.semantic);
                        let w = sim * exp(-dist * 10.0);
                        
                        if (w > 0.001) {
                            accum += src.voltage * w;
                            weight_sum += w;
                            
                            if (sim > max_sim) {
                                max_sim = sim;
                                best_source_idx = src_idx;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Update Activation
    if (weight_sum > 0.001) {
        let input = accum / weight_sum;
        // Temporal smoothing / Leaky integrator
        n.voltage = mix(n.voltage, input, 0.2);
        
        // Plasticity: VSA Learning
        // Adjust semantic hypervector to better match the source pattern
        if (params.train_mode == 1u && best_source_idx != 0xFFFFFFFFu) {
             let src = neurons[best_source_idx];
             let seed = id.x + u32(params.time * 1000.0);
             let mask = vec4<u32>(hash(seed), hash(seed+1u), hash(seed+2u), hash(seed+3u));
             
             // Higher learning rate for Bottleneck (L3) to force abstraction
             let lr_mask = select(0x00010101u, 0x01010101u, n.layer == 3u); 
             let update_mask = mask & vec4<u32>(lr_mask);
             
             // Move towards source semantic
             n.semantic = (n.semantic & ~update_mask) | (src.semantic & update_mask);
        }
    } else {
        n.voltage *= 0.9; // Decay if no input
    }

    neurons[idx] = n;
}

@compute @workgroup_size(64)
fn cs_generate_lines(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    let n = neurons[idx];
    
    // Sparsity threshold
    if (abs(n.voltage) < 0.05) {
        line_buffer[idx*2].pos = vec4<f32>(0.0);
        line_buffer[idx*2+1].pos = vec4<f32>(0.0);
        return;
    }
    
    if (n.layer == 0u) { return; }
    let source_layer = n.layer - 1u;
    let layer_stride = params.grid_dim * params.grid_dim;
    let source_grid_offset = source_layer * layer_stride;

    
    var best_idx = 0xFFFFFFFFu;
    var best_dist = 100.0;
    
    let grid_dim = f32(params.grid_dim);
    let center_u = (n.pos.x + 1.0) * 0.5 * grid_dim;
    let center_v = (n.pos.y + 1.0) * 0.5 * grid_dim;
    
    // Find best connection to draw
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
             let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
             let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
            let g_idx = gx + gy * params.grid_dim + source_grid_offset; 
             let src_idx = atomicLoad(&spatial_grid[g_idx]);
             
             if (src_idx != 0xFFFFFFFFu) {
                 let src = neurons[src_idx];
                 if (src.layer == source_layer) {
                     let d = distance(n.pos, src.pos);
                     let sim = hamming_similarity(n.semantic, src.semantic);
                     
                     // Weight connection strength by proximity and similarity
                     let score = d / (sim + 0.1);
                     
                     if (score < best_dist) {
                         best_dist = score;
                         best_idx = src_idx;
                     }
                 }
             }
        }
    }
    
    if (best_idx != 0xFFFFFFFFu) {
        let src = neurons[best_idx];
        
        // Z-Stack Visualization
        // Map layers 0..6 to -3.0..3.0 range
        let z_curr = f32(n.layer) - 3.0;
        let z_src = f32(src.layer) - 3.0;
        
        // 3.0 multiplier matches vs_main scaling
        let p1 = vec3<f32>(n.pos, z_curr * 0.5) * 3.0;
        let p2 = vec3<f32>(src.pos, z_src * 0.5) * 3.0;
        
        var col = vec4<f32>(1.0);
        
        if (n.layer <= 3u) {
            // Encoder Path (Error propagation): RED
            col = vec4<f32>(1.0, 0.1, 0.1, 0.3); 
        } else {
            // Decoder Path (Prediction generation): CYAN
            col = vec4<f32>(0.0, 1.0, 1.0, 0.3);
        }
        
        line_buffer[idx*2].pos = vec4<f32>(p1, 1.0);
        line_buffer[idx*2].color = col;
        line_buffer[idx*2+1].pos = vec4<f32>(p2, 1.0);
        line_buffer[idx*2+1].color = col;
    } else {
        line_buffer[idx*2].pos = vec4<f32>(0.0);
        line_buffer[idx*2+1].pos = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(8, 8)
fn cs_render_dream(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    let uv = vec2<f32>(f32(id.x) / 512.0, f32(id.y) / 512.0);
    let pos = uv * 2.0 - 1.0;
    
    // Sample L6 neurons (The Predictor Layer)
    var val = 0.0;
    var w_sum = 0.0;
    
    let grid_dim = f32(params.grid_dim);
    let center_u = uv.x * grid_dim;
    let center_v = uv.y * grid_dim;
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
             let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
             let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
             let idx = atomicLoad(&spatial_grid[gx + gy * params.grid_dim]);
             
             if (idx != 0xFFFFFFFFu) {
                 let n = neurons[idx];
                 if (n.layer == 6u) {
                     let d = distance(pos, n.pos);
                     let w = exp(-d * 50.0); // Smooth reconstruction kernel
                     val += n.voltage * w;
                     w_sum += w;
                 }
             }
        }
    }
    
    let final_color = select(0.0, val / w_sum, w_sum > 0.001);
    textureStore(prediction_tex, vec2<i32>(id.xy), vec4<f32>(final_color, final_color, final_color, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_render_error(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    // 1. Calculate position
    let uv = vec2<f32>(f32(id.x) / 512.0, f32(id.y) / 512.0);
    let pos = uv * 2.0 - 1.0;
    
    // 2. Search for L0 (Retina) neurons near this pixel
    var error_accum = 0.0;
    var w_sum = 0.0;
    
    let grid_dim = f32(params.grid_dim);
    let center_u = uv.x * grid_dim;
    let center_v = uv.y * grid_dim;
    
    // Small kernel search to reconstruct the image from neurons
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
             let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
             let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
             
             let idx = atomicLoad(&spatial_grid[gx + gy * params.grid_dim]);
             
             if (idx != 0xFFFFFFFFu) {
                 let n = neurons[idx];
                 // WE WANT LAYER 0 (Retina/Error Layer)
                 if (n.layer == 0u) {
                     let d = distance(pos, n.pos);
                     // L0 is dense, so we use a tight kernel
                     let w = exp(-d * 100.0); 
                     
                     // Voltage in L0 = (Reality - Prediction)
                     // We visualize the absolute error
                     error_accum += n.voltage * w;
                     w_sum += w;
                 }
             }
        }
    }
    
    let val = select(0.0, error_accum / w_sum, w_sum > 0.0001);
    
    // VISUALIZATION:
    // Red = Positive Error (Reality > Prediction)
    // Blue = Negative Error (Prediction > Reality)
    // Black = Perfect Prediction
    var col = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    if (val > 0.0) {
        col = vec4<f32>(val * 5.0, 0.0, 0.0, 1.0); // Amplify for visibility
    } else {
        col = vec4<f32>(0.0, 0.0, abs(val) * 5.0, 1.0);
    }
    
    textureStore(output_tex, vec2<i32>(id.xy), col);
}

// --- RENDERING SHADERS ---

@group(1) @binding(0) var<uniform> camera_uni: Camera;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec3<f32>,
    padding: u32,
    time: f32,
}

@vertex
fn vs_main(@location(0) vertex_pos: vec3<f32>, @builtin(instance_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let n = neurons[idx];
    
    // Stack layers in Z
    // Layer 0 at Z -3.0, Layer 3 at 0.0, Layer 6 at +3.0
    let z_depth = (f32(n.layer) - 3.0) * 0.5;
    
    let world_pos = vec3<f32>(n.pos, z_depth) * 3.0 + vertex_pos * 0.015;
    out.clip_position = camera_uni.view_proj * vec4<f32>(world_pos, 1.0);
    
    var col = vec3<f32>(0.2);
    let val = n.voltage;
    
    if (n.layer == 0u) {
        // L0 (Retina): White = Active, Red = Negative Error
        col = vec3<f32>(val + 0.5, val + 0.5, val + 0.5);
    } else if (n.layer <= 3u) {
        // Encoder: Red intensity (representing Error magnitude)
        col = vec3<f32>(abs(val)*2.0, 0.0, 0.0);
    } else if (n.layer < 6u) {
        // Decoder: Cyan intensity (representing Reconstruction features)
        col = vec3<f32>(0.0, abs(val)*2.0, abs(val)*2.0);
    } else {
        // L6 (Prediction Output): Greenish/White
        col = vec3<f32>(val, val + 0.2, val); 
    }
    
    out.color = vec4<f32>(col, 1.0);
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}

@vertex
fn vs_lines(@location(0) pos: vec4<f32>, @location(1) col: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera_uni.view_proj * vec4<f32>(pos.xyz, 1.0);
    out.color = col;
    return out;
}

@fragment
fn fs_lines(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}