// --- STRUCTS ---

// COMPRESSED NEURON: 48 Bytes (Aligned to 16)
struct Neuron {
    // 128-bit Semantic Hypervector (SDR / VSA Identity)
    semantic: vec4<u32>,  // 16 bytes
    
    // Spatial Position
    pos: vec2<f32>,       // 8 bytes
    
    // Predictive Coding State
    voltage: f32,         // 4 bytes (Prediction Error / Activity)
    prediction: f32,      // 4 bytes (Internal Belief / Mu)
    precision_value: f32,       // 4 bytes (Confidence / Inverse Sigma)
    layer: u32,           // 4 bytes
    
    // 48 Bytes Total. Fits 3 neurons per 144-byte stride roughly, very cache friendly.
    pad0: f32,            // 4 bytes padding to reach 48 bytes alignment check
    pad1: f32,            // 4 bytes (Total 48 bytes now if vec4 aligns)
};

struct SimParams {
    neuron_count: u32,
    time: f32,
    dt: f32,
    grid_dim: u32,
    train_mode: u32, 
    use_camera: u32, 
    // Padding to match C struct alignment
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
// Kernel removed - logic is now intrinsic to VSA/Predictive Coding
@group(0) @binding(2) var<storage, read_write> spatial_grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> line_buffer: array<LineVertex>;

@group(0) @binding(4) var input_tex: texture_2d<f32>;
@group(0) @binding(5) var input_sampler: sampler;
@group(0) @binding(6) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var prediction_tex: texture_storage_2d<rgba32float, write>;

// --- UTILITY ---

// PCG Hash for random generation
fn hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(hash(seed)) / 4294967295.0;
}

// Hamming distance for VSA Similarity (128-bit)
fn hamming_similarity(a: vec4<u32>, b: vec4<u32>) -> f32 {
    let diff = a ^ b;
    let bits = countOneBits(diff.x) + countOneBits(diff.y) + countOneBits(diff.z) + countOneBits(diff.w);
    // Normalize: 0 bits diff = 1.0 sim, 64 bits diff = 0.0 sim
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
    
    let retina_dim = 160u;
    let retina_count = retina_dim * retina_dim;
    
    // Initialize Semantic Hypervector (Random SDR)
    n.semantic = vec4<u32>(hash(seed), hash(seed+1u), hash(seed+2u), hash(seed+3u));
    
    if (idx < retina_count) {
        // Layer 0: Retina
        n.layer = 0u;
        let rx = idx % retina_dim;
        let ry = idx / retina_dim;
        let u = f32(rx) / f32(retina_dim);
        let v = f32(ry) / f32(retina_dim);
        n.pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.precision_value = 1.0; // High confidence in reality
    } else {
        // Layer 1: Cortex
        n.layer = 1u;
        n.pos = vec2<f32>(rand_f32(seed+4u), rand_f32(seed+5u)) * 2.0 - 1.0;
        n.precision_value = 0.1; // Low initial confidence
    }
    
    n.voltage = 0.0;
    n.prediction = 0.0;
    
    neurons[idx] = n;
}

@compute @workgroup_size(64)
fn cs_clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&spatial_grid)) { return; }
    atomicStore(&spatial_grid[idx], 0xFFFFFFFFu);
}

@compute @workgroup_size(64)
fn cs_populate_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    // Simple spatial hashing for neighbor lookup
    let grid_idx = hash_to_grid(neurons[idx].pos);
    atomicStore(&spatial_grid[grid_idx], idx);
}

@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let post_idx = id.x;
    if (post_idx >= params.neuron_count) { return; }
    
    var post = neurons[post_idx];
    
    // ---------------------------------------------------------
    // RETINA (Layer 0): Input Processing
    // ---------------------------------------------------------
    if (post.layer == 0u) {
        let uv = (post.pos + 1.0) * 0.5;
        var reality = 0.0;
        
        if (params.use_camera == 1u) {
            reality = textureSampleLevel(input_tex, input_sampler, uv, 0.0).r;
        } else {
            // Dream Mode
            let t = params.time;
            let warp_uv = uv + vec2<f32>(sin(uv.y * 10.0 + t), cos(uv.x * 10.0 + t)) * 0.01;
            reality = textureSampleLevel(input_tex, input_sampler, warp_uv, 0.0).r;
            reality += (rand_f32(post_idx + u32(t * 60.0)) - 0.5) * 0.1; // Noise
        }
        
        // Predictive Coding: Error = (Reality - Prediction) * Precision
        // Retina predictions come from Top-Down feedback (not implemented fully here for simplicity, 
        // so we just use reality as the driver)
        let val = (reality - 0.5) * 2.0;
        post.prediction = val;
        post.voltage = val; // Retina transmits input directly as "error" to cortex
        
        neurons[post_idx] = post;
        return;
    }

    // ---------------------------------------------------------
    // CORTEX (Layer 1): VSA Integration & Predictive Learning
    // ---------------------------------------------------------
    
    var input_accum = 0.0;
    var weight_accum = 0.0;
    
    var best_neighbor_idx = 0xFFFFFFFFu;
    var max_sim = 0.0;

    // 1. PROCEDURAL CONNECTIVITY (Implicit Synapses)
    // Tile Search: 4x4 area around neuron
    // We look for neighbors in the spatial grid. 
    // "Connection" exists if they are spatially close AND semantically similar.
    
    let grid_dim = f32(params.grid_dim);
    let center_u = (post.pos.x + 1.0) * 0.5 * grid_dim;
    let center_v = (post.pos.y + 1.0) * 0.5 * grid_dim;
    
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
            let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
            let g_idx = gx + gy * params.grid_dim;
            
            let pre_idx = atomicLoad(&spatial_grid[g_idx]);
            
            if (pre_idx != 0xFFFFFFFFu && pre_idx != post_idx) {
                let pre = neurons[pre_idx];
                
                // 1. Spatial Constraint
                let d_vec = post.pos - pre.pos;
                let d2 = dot(d_vec, d_vec);
                
                if (d2 < 0.05) { // Radius
                    let spatial_w = exp(-d2 * 40.0);
                    
                    // 2. Semantic Constraint (VSA Binding)
                    // How similar are the hypervectors?
                    let semantic_w = hamming_similarity(post.semantic, pre.semantic);
                    
                    // Combined Weight
                    // Boost semantic influence
                    let w = spatial_w * (semantic_w * semantic_w * semantic_w + 0.01); 
                    
                    if (w > 0.001) {
                        // Integrate Input (Weighted by Pre Activity)
                        // If Pre is Retina, use voltage. If Cortex, use voltage (Error) or prediction?
                        // Standard PC: Cortex learns from Retina Error.
                        input_accum += pre.voltage * w;
                        weight_accum += w;
                        
                        if (semantic_w > max_sim && pre.layer == 0u) { // Prefer locking to input
                            max_sim = semantic_w;
                            best_neighbor_idx = pre_idx;
                        }
                    }
                }
            }
        }
    }
    
    // Normalize Input
    let input_signal = select(0.0, input_accum / weight_accum, weight_accum > 0.0);
    
    // 2. PREDICTIVE CODING UPDATE
    // Update internal belief (mu) based on input error
    let learning_rate = 0.05;
    post.prediction = mix(post.prediction, input_signal, learning_rate * post.precision_value);
    
    // Calculate own Output Error (Voltage)
    // In this simplified model, Cortex Voltage = Prediction (Activity)
    post.voltage = post.prediction;
    
    // Update Precision (Sigma) - Become more confident if input is stable
    let surprise = abs(post.prediction - input_signal);
    post.precision_value = mix(post.precision_value, 1.0 / (surprise + 0.1), 0.01);

    // 3. HYPERVECTOR LEARNING (The VSA Magic)
    // If we have a strong input source but low semantic match, rotate our vector towards it.
    // This simulates Hebbian learning of connections without storing weights.
    if (params.train_mode == 1u && best_neighbor_idx != 0xFFFFFFFFu) {
        let best_pre = neurons[best_neighbor_idx];
        
        // If we are active and neighbor is active
        if (abs(post.voltage) > 0.1 && abs(best_pre.voltage) > 0.1) {
             // Stochastic bit flipping to align vectors
             let diff = post.semantic ^ best_pre.semantic;
             let mask = vec4<u32>(
                 hash(id.x + u32(params.time * 1000.0)),
                 hash(id.x + u32(params.time * 2000.0)),
                 hash(id.x + u32(params.time * 3000.0)),
                 hash(id.x + u32(params.time * 4000.0))
             );
             
             // 1% chance to flip bits to match parent
             let update_mask = mask & vec4<u32>(0x01010101u); 
             post.semantic = (post.semantic & ~update_mask) | (best_pre.semantic & update_mask);
        }
    }

    neurons[post_idx] = post;

    // 4. VISUALIZATION
    let cortex_uv = (post.pos + 1.0) * 0.5;
    let tx = i32(cortex_uv.x * 512.0);
    let ty = i32(cortex_uv.y * 512.0);
    if (tx >= 0 && tx < 512 && ty >= 0 && ty < 512) {
        let act = post.voltage;
        let col = vec4<f32>(
            max(0.0, act), 
            post.precision_value * 0.5, 
            max(0.0, -act), 
            1.0
        );
        textureStore(output_tex, vec2<i32>(tx, ty), col);
    }
}

@compute @workgroup_size(64)
fn cs_generate_lines(@builtin(global_invocation_id) id: vec3<u32>) {
    // PROCEDURAL LINE GENERATION
    // Since we don't store synapses, we re-discover the strongest link to visualize it.
    
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    let n = neurons[idx];
    
    // Only visualize Cortex neurons with high activity
    if (n.layer == 0u || abs(n.voltage) < 0.1) { 
        // Clear line
        line_buffer[idx*2].pos = vec4<f32>(0.0);
        line_buffer[idx*2+1].pos = vec4<f32>(0.0);
        return; 
    }

    var best_pre_idx = 0xFFFFFFFFu;
    var best_w = 0.0;
    
    // Quick local search (smaller radius than update for speed)
    let grid_dim = f32(params.grid_dim);
    let center_u = (n.pos.x + 1.0) * 0.5 * grid_dim;
    let center_v = (n.pos.y + 1.0) * 0.5 * grid_dim;
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
            let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
            let g_idx = gx + gy * params.grid_dim;
            
            let pre_idx = atomicLoad(&spatial_grid[g_idx]);
            if (pre_idx != 0xFFFFFFFFu && pre_idx != idx) {
                 let pre = neurons[pre_idx];
                 // Visualization prefers Semantic Similarity
                 let sim = hamming_similarity(n.semantic, pre.semantic);
                 let dist = distance(n.pos, pre.pos);
                 let w = sim / (dist + 0.01);
                 
                 if (w > best_w) {
                     best_w = w;
                     best_pre_idx = pre_idx;
                 }
            }
        }
    }
    
    if (best_pre_idx != 0xFFFFFFFFu) {
        let target_neuron = neurons[best_pre_idx];
        let origin = vec3<f32>(n.pos, 0.0) * 3.0;
        let dest = vec3<f32>(target_neuron.pos, select(0.0, -0.8, target_neuron.layer==0u)) * 3.0;
        
        let alpha = clamp(best_w * 0.5, 0.1, 0.8);
        let col = vec3<f32>(0.0, 1.0, 0.5); // Cyan lines for VSA links
        
        line_buffer[idx*2].pos = vec4<f32>(origin, 1.0);
        line_buffer[idx*2].color = vec4<f32>(col, alpha);
        line_buffer[idx*2+1].pos = vec4<f32>(dest, 1.0);
        line_buffer[idx*2+1].color = vec4<f32>(col, alpha);
    } else {
        line_buffer[idx*2].pos = vec4<f32>(0.0);
        line_buffer[idx*2+1].pos = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(8, 8)
fn cs_render_dream(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    let uv = vec2<f32>(id.xy) / vec2<f32>(dim);
    let rx = u32(uv.x * 160.0);
    let ry = u32(uv.y * 160.0);
    let idx = rx + ry * 160u;
    
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (idx < params.neuron_count) {
        let n = neurons[idx];
        let v = n.voltage * 0.5 + 0.5;
        color = vec4<f32>(v, v, v, 1.0);
    }
    textureStore(prediction_tex, vec2<i32>(id.xy), color);
}

@compute @workgroup_size(8, 8)
fn cs_clear_cortex(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

// --- RENDERING ---

@group(1) @binding(0) var<uniform> camera_uni: Camera;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec3<f32>,
    padding: u32,
    time: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) vertex_pos: vec3<f32>, @builtin(instance_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let n = neurons[idx];
    
    let z_offset = select(0.0, -0.8, n.layer == 0u);
    let world_pos = vec3<f32>(n.pos, z_offset) * 3.0 + vertex_pos * 0.015;
    
    out.clip_position = camera_uni.view_proj * vec4<f32>(world_pos, 1.0);
    
    var col = vec3<f32>(0.2);
    if (n.layer == 0u) {
        col = vec3<f32>(n.voltage + 0.5);
    } else {
        let energy = abs(n.voltage);
        col = vec3<f32>(energy, n.precision_value * 0.5, 0.2);
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