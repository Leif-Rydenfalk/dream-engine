// --- DATA STRUCTURES ---

struct Neuron {
    // xy = uv coordinates (0.0-1.0)
    pos_physical: vec4<f32>,

    // xyz = The RGB Color / Concept this neuron represents
    // w = clustering radius
    pos_semantic: vec4<f32>,

    // x = current voltage
    // y = last prediction
    // z = error trace
    state: vec4<f32>,
    
    // Momentum
    velocity: vec4<f32>,
}

struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    train_mode: u32, // 1 = Training, 0 = Dreaming
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

// --- BINDINGS ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var input_texture: texture_2d<f32>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<storage, read_write> spatial_grid: array<atomic<u32>>; 

const GRID_DIM: u32 = 64u;
const CELL_SIZE: f32 = 0.1; // Tighter clustering for color precision

// --- UTILS ---

fn hash_coords(pos: vec3<f32>) -> u32 {
    // Map RGB (0.0-1.0) to Grid Indices
    // We assume pos_semantic is roughly 0.0 to 1.0 (Color space)
    let scaled = pos * (1.0 / CELL_SIZE); 
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
    // Hash based on my Concept (Color/Position)
    let grid_idx = hash_coords(me.pos_semantic.xyz);
    atomicStore(&spatial_grid[grid_idx], idx);
}

// --- PASS 3: UPDATE NEURONS (THE BRAIN) ---
@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }

    var me = neurons[idx];

    // ==========================================
    // PHASE 1: THE DREAM (Monte Carlo Sampling)
    // ==========================================
    
    var neighbor_activity_sum = 0.0;
    var connection_count = 0.0;
    
    // Self-Memory (The "Ghosting")
    neighbor_activity_sum += me.state.x * 0.9;
    connection_count += 1.0;

    // We will sample 8 random locations around my concept
    // effectively "casting a net" to catch nearby neurons
    for (var i = 0u; i < 8u; i++) {
        
        // Generate a random offset based on ID and Iteration
        let jitter = rand_vec3(idx + i * 1000u + u32(params.time * 100.0)) * CELL_SIZE;
        let probe_pos = me.pos_semantic.xyz + jitter;
        
        let grid_idx = hash_coords(probe_pos);
        let neighbor_id = atomicLoad(&spatial_grid[grid_idx]);
        
        if (neighbor_id != 0xFFFFFFFFu && neighbor_id != idx) {
            let other = neurons[neighbor_id];
            
            // Are they close enough in color/concept?
            let dist = distance(me.pos_semantic.xyz, other.pos_semantic.xyz);
            
            if (dist < CELL_SIZE) {
                // Synaptic Weight: Closer = Stronger
                let weight = 1.0 - (dist / CELL_SIZE);
                neighbor_activity_sum += other.state.x * weight;
                connection_count += weight;
            }
        }
    }

    // Average the input
    if (connection_count > 0.0) {
        neighbor_activity_sum /= connection_count; // Normalize
    }
    
    // Amplify slightly (Gain Control) so the signal doesn't die
    let predicted_activation = clamp(neighbor_activity_sum * 1.1, 0.0, 1.0);

    // ==========================================
    // PHASE 2: VISUALIZATION (Write Output)
    // ==========================================
    // We write the PREDICTION to the texture. 
    // If the network is wrong, you will see it be wrong here.
    
    if (predicted_activation > 0.01) {
        let dims = textureDimensions(output_texture);
        let uv = me.pos_physical.xy * 0.5 + 0.5; // Map -1..1 to 0..1
        let out_uv = vec2<i32>(vec2<f32>(dims) * uv);
        
        // The color is my Semantic Concept (what I believe I am)
        // The brightness is my Predicted Activation (how sure I am)
        let pixel_color = abs(me.pos_semantic.xyz) * predicted_activation;
        
        textureStore(output_texture, out_uv, vec4<f32>(pixel_color, 1.0));
    }

    // ==========================================
    // PHASE 3: REALITY CHECK (Training)
    // ==========================================
    
    var error = 0.0;
    var input_activation = 0.0;

    if (params.train_mode == 1u) {
        // Now we open our eyes.
        let uv = me.pos_physical.xy * 0.5 + 0.5;
        let real_pixel = textureSampleLevel(input_texture, input_sampler, uv, 0.0);
        
        // How active should I have been?
        // If my concept (Color) matches the real pixel, I should be active.
        let color_match = 1.0 - distance(real_pixel.rgb, abs(me.pos_semantic.xyz));
        input_activation = clamp(color_match, 0.0, 1.0);

        // Error = Difference between Dream and Reality
        error = abs(predicted_activation - input_activation);
        
        // ==========================================
        // PHASE 4: LEARNING (Neuroplasticity)
        // ==========================================
        let learning_rate = 0.05;

        if (error > 0.2) {
            // TERROR: I was wrong.
            // My Concept (Pos Semantic) doesn't match the reality at this Physical Location.
            // MOVE my concept towards the real color.
            
            let color_diff = real_pixel.rgb - abs(me.pos_semantic.xyz);
            
            // Physically move towards the correct color cluster
            me.velocity.x += color_diff.r * learning_rate;
            me.velocity.y += color_diff.g * learning_rate;
            me.velocity.z += color_diff.b * learning_rate;
            
            // Jitter slightly to break out of bad clusters
            me.velocity += (vec4<f32>(rand_vec3(idx), 0.0) * 0.01);
        } else {
            // DOPAMINE: I was right.
            // Crystallize (stop moving).
            me.velocity *= 0.5; 
        }
        
        // Force state sync for next frame (Hebbian Imprinting)
        // me.state.x = mix(predicted_activation, input_activation, 0.5);
        me.state.x = mix(predicted_activation, input_activation, 0.001);
        
    } else {
        // DREAM MODE: Closed Loop.
        // State purely follows prediction.
        me.state.x = predicted_activation;
        error = 0.0;
    }

    // Physics Update
    me.pos_semantic += me.velocity * 0.1;
    me.velocity *= 0.9; // Friction
    
    // Bound check (Concepts stay within 0-1 RGB space roughly)
    me.pos_semantic = clamp(me.pos_semantic, vec4<f32>(-1.0), vec4<f32>(1.0));

    me.state.y = predicted_activation;
    me.state.z = error;
    neurons[idx] = me;
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
    @location(4) pos_semantic: vec4<f32>,
    @location(5) state: vec4<f32>,
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
    
    // Visualize the RGB Color Cube
    // x = Red, y = Green, z = Blue
    // We scale it up to be visible
    let world_pos = (instance.pos_semantic.xyz * 10.0) + (model.position * 0.05);
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Color the particle by its Concept
    let base_color = abs(instance.pos_semantic.xyz);
    let activity = instance.state.x;
    
    // Dim inactive neurons, Light up active ones
    out.color = base_color * (0.2 + activity * 1.5);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}