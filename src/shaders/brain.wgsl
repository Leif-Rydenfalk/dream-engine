// --- STRUCTS ---

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// COMPRESSED NEURON: 48 Bytes
struct Neuron {
    // 128-bit Semantic Hypervector (SDR Identity)
    semantic: vec4<u32>,  // 16 bytes
    
    // Spatial Position
    pos: vec2<f32>,       // 8 bytes
    
    // Predictive Coding State
    voltage: f32,         // 4 bytes (Error)
    prediction: f32,      // 4 bytes (Prediction/Mu)
    precision_value: f32,       // 4 bytes (Precision/Inverse Sigma)
    layer: u32,           // 4 bytes
    
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
@group(0) @binding(7) var prediction_tex: texture_storage_2d<rgba32float, write>; // Write-only for Dream Pass
@group(0) @binding(8) var prediction_tex_read: texture_2d<f32>;                // Read-only for Update Pass

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
    
    // Init Semantic Hypervector (Random SDR)
    n.semantic = vec4<u32>(hash(seed), hash(seed+1u), hash(seed+2u), hash(seed+3u));
    
    // Layer Distribution
    // L0: Retina (25,600)
    // L1: V1 (~43k)
    // L2: V2 (~43k)
    // L3: V4 (~43k)
    // L4: IT (~43k)
    
    if (idx < 25600u) {
        n.layer = 0u; // Retina
        let dim = 160u;
        let rx = idx % dim;
        let ry = idx / dim;
        let u = f32(rx) / f32(dim);
        let v = f32(ry) / f32(dim);
        n.pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.precision_value = 10.0; 
    } else {
        if (idx < 69200u) { n.layer = 1u; }
        else if (idx < 112800u) { n.layer = 2u; }
        else if (idx < 156400u) { n.layer = 3u; }
        else { n.layer = 4u; }
        
        // Cortex positions are random but can be topologically organized by seed
        // To allow local connections, we map them roughly to screen space but with jitter
        n.pos = vec2<f32>(rand_f32(seed+4u), rand_f32(seed+5u)) * 2.0 - 1.0;
        n.precision_value = 1.0;
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
    let grid_idx = hash_to_grid(neurons[idx].pos);
    atomicStore(&spatial_grid[grid_idx], idx);
}

@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let post_idx = id.x;
    if (post_idx >= params.neuron_count) { return; }
    
    var post = neurons[post_idx];
    
    // ---------------------------------------------------------
    // LAYER 0: RETINA (Sensor)
    // ---------------------------------------------------------
    if (post.layer == 0u) {
        let uv = (post.pos + 1.0) * 0.5;
        
        // 1. Sample Reality (Input)
        var reality = 0.0;
        if (params.use_camera == 1u) {
            reality = textureSampleLevel(input_tex, input_sampler, uv, 0.0).r;
        } else {
            // Dream mode noise/drift
            let t = params.time;
            let warp = uv + vec2<f32>(sin(uv.y*4.0+t), cos(uv.x*4.0+t)) * 0.02;
            reality = textureSampleLevel(input_tex, input_sampler, warp, 0.0).r;
            reality += (rand_f32(post_idx + u32(t * 60.0)) - 0.5) * 0.1;
        }
        
        // 2. Sample Prediction (Feedback from Cortex)
        // READ from the read-only binding
        let top_down_pred = textureLoad(prediction_tex_read, vec2<i32>(uv * 512.0), 0).r;
        
        // 3. Compute Error (Voltage)
        // Voltage = Reality - Prediction
        post.voltage = (reality - top_down_pred) * 2.0;
        post.prediction = top_down_pred; // Store what we thought
        
        neurons[post_idx] = post;
        return;
    }

    // ---------------------------------------------------------
    // LAYERS 1-4: CORTEX (Predictive Processing)
    // ---------------------------------------------------------
    
    var driver_accum = 0.0; // Feedforward (L-1)
    var driver_weight = 0.0;
    
    var context_accum = 0.0; // Lateral (L)
    var context_weight = 0.0;
    
    var best_driver_idx = 0xFFFFFFFFu;
    var max_sim = 0.0;

    let grid_dim = f32(params.grid_dim);
    let center_u = (post.pos.x + 1.0) * 0.5 * grid_dim;
    let center_v = (post.pos.y + 1.0) * 0.5 * grid_dim;
    
    // Search Neighborhood
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let gx = u32(clamp(center_u + f32(dx), 0.0, grid_dim - 1.0));
            let gy = u32(clamp(center_v + f32(dy), 0.0, grid_dim - 1.0));
            let g_idx = gx + gy * params.grid_dim;
            
            let pre_idx = atomicLoad(&spatial_grid[g_idx]);
            
            if (pre_idx != 0xFFFFFFFFu && pre_idx != post_idx) {
                let pre = neurons[pre_idx];
                
                // Spatial filter
                let d_vec = post.pos - pre.pos;
                let d2 = dot(d_vec, d_vec);
                
                if (d2 < 0.06) {
                    let spatial_w = exp(-d2 * 30.0);
                    let semantic_w = hamming_similarity(post.semantic, pre.semantic);
                    
                    // Determine Role based on Layer
                    // Feedforward: L-1 -> L
                    if (pre.layer == post.layer - 1u) {
                        let w = spatial_w * (semantic_w * semantic_w + 0.01);
                        if (w > 0.001) {
                            // Driver sends Error (Voltage)
                            driver_accum += pre.voltage * w;
                            driver_weight += w;
                            
                            if (semantic_w > max_sim) {
                                max_sim = semantic_w;
                                best_driver_idx = pre_idx;
                            }
                        }
                    }
                    // Lateral: L -> L
                    else if (pre.layer == post.layer) {
                        let w = spatial_w * (semantic_w + 0.01);
                        if (w > 0.001) {
                            // Context sends Activity/Belief (Prediction)
                            context_accum += pre.prediction * w;
                            context_weight += w;
                        }
                    }
                }
            }
        }
    }
    
    // Normalize Signals
    let driver_in = select(0.0, driver_accum / driver_weight, driver_weight > 0.0);
    let context_in = select(0.0, context_accum / context_weight, context_weight > 0.0);
    
    // ---------------------------------------------------------
    // PREDICTIVE CODING UPDATE RULE
    // ---------------------------------------------------------
    
    // 1. Update Prediction (Internal Belief)
    // Driven by Context (Lateral) and previous state
    let alpha = 0.1 * post.precision_value;
    post.prediction = mix(post.prediction, context_in, alpha);
    
    // 2. Update Voltage (Prediction Error)
    // Error = Input (from below) - Prediction (from self/lateral)
    // This Error propagates up to the next layer
    post.voltage = driver_in - post.prediction;
    
    // 3. Update Precision
    // Inverse variance of the error
    let error_mag = abs(post.voltage);
    post.precision_value = mix(post.precision_value, 1.0 / (error_mag + 0.1), 0.05);

    // ---------------------------------------------------------
    // VSA LEARNING (Plasticity)
    // ---------------------------------------------------------
    if (params.train_mode == 1u && best_driver_idx != 0xFFFFFFFFu) {
        let driver = neurons[best_driver_idx];
        
        // If we have significant error to explain
        if (error_mag > 0.1) {
            // "Transitions" learning: Move towards (Driver XOR Context)
            // If context is weak, just move towards Driver
            
            let target_semantic = driver.semantic;
            // Simple bit-flip plasticity towards the driver that caused the error
            // This aligns the RF to explain the input
            
            let diff = post.semantic ^ target_semantic;
            let seed = id.x + u32(params.time * 1000.0);
            let mask = vec4<u32>(
                 hash(seed), hash(seed+1u), hash(seed+2u), hash(seed+3u)
            );
            
            // 1% learning rate
            let update_mask = mask & vec4<u32>(0x01010101u);
            post.semantic = (post.semantic & ~update_mask) | (target_semantic & update_mask);
        }
    }

    neurons[post_idx] = post;

    // Visualization
    let cortex_uv = (post.pos + 1.0) * 0.5;
    let tx = i32(cortex_uv.x * 512.0);
    let ty = i32(cortex_uv.y * 512.0);
    if (tx >= 0 && tx < 512 && ty >= 0 && ty < 512) {
        let v = post.voltage; // Show error
        // Red = Positive Error, Blue = Negative Error, Green = Prediction
        let col = vec4<f32>(max(0.0, v), abs(post.prediction)*0.5, max(0.0, -v), 1.0);
        textureStore(output_tex, vec2<i32>(tx, ty), col);
    }
}

@compute @workgroup_size(64)
fn cs_generate_lines(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    let n = neurons[idx];
    
    // Only draw lines for active neurons
    if (abs(n.voltage) < 0.1) {
        line_buffer[idx*2].pos = vec4<f32>(0.0);
        line_buffer[idx*2+1].pos = vec4<f32>(0.0);
        return;
    }
    
    var best_pre_idx = 0xFFFFFFFFu;
    var best_w = 0.0;
    
    // Visualize the strongest FEEDFORWARD connection
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
                 if (pre.layer == n.layer - 1u) { // Visualize Drivers
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
    }
    
    if (best_pre_idx != 0xFFFFFFFFu) {
        let target_value = neurons[best_pre_idx];
        // Stack layers in Z for visualization
        // Layer 0 at z=-1.0, Layer 4 at z=1.0
        let z_post = (f32(n.layer) * 0.5) - 1.0;
        let z_pre = (f32(target_value.layer) * 0.5) - 1.0;
        
        let origin = vec3<f32>(n.pos, z_post) * 3.0;
        let dest = vec3<f32>(target_value.pos, z_pre) * 3.0;
        
        let alpha = clamp(best_w * 0.5, 0.1, 0.5);
        // Cyan lines
        let col = vec3<f32>(0.0, 1.0, 0.8); 
        
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
    
    // We can now safely write to binding 7 (prediction_tex)
    // because this runs in a separate pass from the reader.
    
    textureStore(prediction_tex, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_clear_cortex(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

// --- RENDERING VERTEX SHADER ---

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
    
    // Stack layers in Z: -1.0 to 1.0
    let z_depth = (f32(n.layer) * 0.5) - 1.0;
    
    let world_pos = vec3<f32>(n.pos, z_depth) * 3.0 + vertex_pos * 0.015;
    out.clip_position = camera_uni.view_proj * vec4<f32>(world_pos, 1.0);
    
    var col = vec3<f32>(0.2);
    
    if (n.layer == 0u) {
        // Retina: White/Black
        let v = n.voltage + 0.5;
        col = vec3<f32>(v, v, v);
    } else {
        // Cortex: Color by Precision/Error
        let err = abs(n.voltage);
        let pred = n.prediction;
        // Red = Error, Green = Pred, Blue = Layer ID
        col = vec3<f32>(err, max(0.0, pred), f32(n.layer)*0.2);
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