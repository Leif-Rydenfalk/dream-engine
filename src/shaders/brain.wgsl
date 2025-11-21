// --- DATA STRUCTURES ---

struct Neuron {
    // xy = uv coordinates (0.0-1.0)
    pos_physical: vec4<f32>,

    // 16-Dimensional Synaptic Weight Vector
    // Stored as 4 vec4s
    weights_a: vec4<f32>, // .xyz also used for 3D position in visualizer
    weights_b: vec4<f32>,
    weights_c: vec4<f32>,
    weights_d: vec4<f32>,

    // x = current voltage (firing rate)
    // y = last prediction
    // z = error trace
    state: vec4<f32>,
}

struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    
    train_mode: u32,
    sample_count: u32,
    mix_rate: f32,
    learning_rate: f32,
    
    dream_decay: f32,
    terror_threshold: f32, 
    _pad1: f32,
    _pad2: f32,
}

// --- BINDINGS ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var input_texture: texture_2d<f32>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<storage, read_write> spatial_grid: array<atomic<u32>>; 

const GRID_DIM: u32 = 64u;
// Smaller cell size because the "Concept Space" (weights_a) is tighter
const CELL_SIZE: f32 = 0.05; 

// --- UTILS ---

fn hash_coords(pos: vec3<f32>) -> u32 {
    // Map Weights (Concept Space) to Grid Indices
    // Normalize typical weight range (-1.0 to 1.0) to grid
    let scaled = (pos + 1.0) * 0.5 * (f32(GRID_DIM)); 
    let grid_pos = vec3<u32>(clamp(scaled, vec3<f32>(0.0), vec3<f32>(f32(GRID_DIM) - 1.0)));
    return grid_pos.x + (grid_pos.y * GRID_DIM) + (grid_pos.z * GRID_DIM * GRID_DIM);
}

// --- PASS 1: CLEAR GRID ---
@compute @workgroup_size(64)
fn cs_clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&spatial_grid)) { return; }
    atomicStore(&spatial_grid[idx], 0xFFFFFFFFu);
}

// --- PASS 2: POPULATE GRID ---
@compute @workgroup_size(64)
fn cs_populate_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }
    let me = neurons[idx];
    // Spatial indexing based on the first 3 dimensions of the weight vector
    // Neurons with similar leading weights will cluster together
    let grid_idx = hash_coords(me.weights_a.xyz);
    atomicStore(&spatial_grid[grid_idx], idx);
}

// --- PASS 3: UPDATE NEURONS (THE BRAIN) ---
@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }

    var me = neurons[idx];

    // --- 1. GATHER INPUT (16 Dimensions) ---
    // Sample a 2x2 block of pixels around the physical location
    let uv = me.pos_physical.xy * 0.5 + 0.5;
    let tex_dims = vec2<f32>(textureDimensions(input_texture));
    let one_pixel = 1.0 / tex_dims;

    // Read 4 pixels (4 channels each = 16 inputs)
    let in_a = textureSampleLevel(input_texture, input_sampler, uv, 0.0);
    let in_b = textureSampleLevel(input_texture, input_sampler, uv + vec2<f32>(one_pixel.x, 0.0), 0.0);
    let in_c = textureSampleLevel(input_texture, input_sampler, uv + vec2<f32>(0.0, one_pixel.y), 0.0);
    let in_d = textureSampleLevel(input_texture, input_sampler, uv + one_pixel, 0.0);

    // --- 2. CALCULATE ACTIVATION (Dot Product) ---
    // "Does the input match my synaptic weights?"
    var activation_energy = 0.0;
    activation_energy += dot(in_a, me.weights_a);
    activation_energy += dot(in_b, me.weights_b);
    activation_energy += dot(in_c, me.weights_c);
    activation_energy += dot(in_d, me.weights_d);
    
    // Apply non-linearity (ReLU-ish)
    activation_energy = max(0.0, activation_energy);
    
    // --- 3. LATERAL CONNECTIONS (The "Dream" layer) ---
    var neighbor_input = 0.0;
    var connection_count = 0.0;
    
    // Self-Memory
    neighbor_input += me.state.x * params.dream_decay; 
    connection_count += 1.0;

    let loops = min(params.sample_count, 32u);
    for (var i = 0u; i < loops; i++) {
        // Jitter in concept space
        let jitter = rand_vec3(idx + i * 1000u + u32(params.time * 60.0)) * CELL_SIZE;
        let probe_pos = me.weights_a.xyz + jitter;
        let grid_idx = hash_coords(probe_pos);
        
        let neighbor_id = atomicLoad(&spatial_grid[grid_idx]);
        if (neighbor_id != 0xFFFFFFFFu && neighbor_id != idx) {
            let other = neurons[neighbor_id];
            // Distance in concept space (similarity)
            let dist = distance(me.weights_a.xyz, other.weights_a.xyz);
            if (dist < CELL_SIZE) {
                // If concepts are similar, they excite each other
                neighbor_input += other.state.x; 
                connection_count += 1.0;
            }
        }
    }
    
    if (connection_count > 0.0) { neighbor_input /= connection_count; }

    // --- 4. INTEGRATE STATE ---
    // Prediction = what I thought would happen (from memory/neighbors)
    let predicted_activation = neighbor_input; 
    
    // Reality = what actually hit my sensors
    let sensory_activation = activation_energy * 0.25; // Scale down raw dot product

    var final_activation = 0.0;
    var error = 0.0;

    if (params.train_mode == 1u) {
        // TRAINING: Reality dominates, but we check surprise
        error = abs(sensory_activation - predicted_activation);
        
        // Mix reality and dream based on how sure we are
        final_activation = mix(predicted_activation, sensory_activation, params.mix_rate);
        
        // --- 5. LEARNING (Surprise-Gated Hebbian) ---
        // If we are surprised (error > threshold), we learn FAST.
        // If prediction matched reality, we don't change much (stability).
        
        var effective_lr = params.learning_rate;
        if (error > params.terror_threshold) {
            effective_lr *= 5.0; // Panic learn!
        }
        
        // Move weights towards the input pattern that caused firing
        // Normalized Oja's rule approximation to prevent weight explosion
        if (final_activation > 0.01) {
            me.weights_a = mix(me.weights_a, in_a, effective_lr * final_activation);
            me.weights_b = mix(me.weights_b, in_b, effective_lr * final_activation);
            me.weights_c = mix(me.weights_c, in_c, effective_lr * final_activation);
            me.weights_d = mix(me.weights_d, in_d, effective_lr * final_activation);
            
            // Normalization (Keep weights in -1..1 range roughly)
            me.weights_a = clamp(me.weights_a, vec4<f32>(-1.0), vec4<f32>(1.0));
            me.weights_b = clamp(me.weights_b, vec4<f32>(-1.0), vec4<f32>(1.0));
            me.weights_c = clamp(me.weights_c, vec4<f32>(-1.0), vec4<f32>(1.0));
            me.weights_d = clamp(me.weights_d, vec4<f32>(-1.0), vec4<f32>(1.0));
        }

    } else {
        // DREAMING: Input is cut off, run purely on internal recurrence
        final_activation = predicted_activation;
        error = 0.0;
    }

    // Store state
    me.state.x = clamp(final_activation, 0.0, 1.0);
    me.state.y = predicted_activation;
    me.state.z = error;
    
    neurons[idx] = me;

    // --- 6. VISUALIZATION OUTPUT ---
    if (me.state.x > 0.1) {
        let dims = textureDimensions(output_texture);
        let out_uv = vec2<i32>(vec2<f32>(dims) * uv);
        // Color = The first 3 weights (The "Concept" of the neuron)
        let pixel_color = abs(me.weights_a.xyz) * me.state.x;
        textureStore(output_texture, out_uv, vec4<f32>(pixel_color, 1.0));
    }
}

fn rand_vec3(seed: u32) -> vec3<f32> {
    let t = f32(seed) + params.time;
    return vec3<f32>(
        fract(sin(t * 12.9898) * 43758.5453),
        fract(sin(t * 78.233) * 43758.5453),
        fract(sin(t * 151.7182) * 43758.5453)
    ) * 2.0 - 1.0;
}

// --- RENDER SHADER ---

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct InstanceInput {
    @location(3) pos_physical: vec4<f32>,
    @location(4) weights_a: vec4<f32>, // Used as Position
    @location(5) weights_b: vec4<f32>, // Used as Color base
    @location(6) weights_c: vec4<f32>,
    @location(7) weights_d: vec4<f32>,
    @location(8) state: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    pos: vec3<f32>,
    _padding: u32,
    time: f32,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    
    // VISUALIZE:
    // We project the 16-D neuron into 3D using its first 3 weights.
    // This creates the "Concept Cloud".
    
    let world_pos = (instance.weights_a.xyz * 8.0) + (model.position * 0.03);
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Color logic:
    // Base color = weights_a (Red/Green/Blue primary concept)
    // Tint = weights_b (Secondary features)
    // Brightness = state.x (Voltage)
    
    let base_color = abs(instance.weights_a.xyz);
    let activity = instance.state.x;
    let error = instance.state.z;
    
    // Flash white/red on error (terror)
    let flash = vec3<f32>(1.0, 0.0, 0.0) * error * 5.0;
    
    out.color = (base_color * (0.1 + activity * 2.0)) + flash;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}