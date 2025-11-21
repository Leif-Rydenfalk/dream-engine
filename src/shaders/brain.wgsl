// --- DATA STRUCTURES ---

struct Neuron {
    // Coordinate Set 1: Fixed Physical (Where it is on screen/tissue)
    // xy = uv coordinates (0.0-1.0), z = depth layer, w = unused
    pos_physical: vec4<f32>,

    // Coordinate Set 2: Dynamic Semantic (The "Weight" / "Concept")
    // xyz = concept space position, w = "mass" or clustering radius
    pos_semantic: vec4<f32>,

    // State
    // x = voltage (current activity)
    // y = prediction_val (what I think the pixel should be)
    // z = error_trace (accumulated surprise)
    // w = unused
    state: vec4<f32>,
    
    // Momentum for semantic movement
    velocity: vec4<f32>,
}

struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    mouse_x: f32,
    mouse_y: f32,
    is_clicking: u32,
    train_mode: u32, // 1 = Training (Input On), 0 = Dreaming (Input Off)
}

// --- BINDINGS ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var input_texture: texture_2d<f32>;       // Camera Feed
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba32float, write>;
// The Spatial Hash Grid (Index -> Neuron ID)
// We use atomic exchange to populate it. Last neuron to write wins.
// This provides stochastic sampling of neighbors, which prevents N^2.
@group(0) @binding(5) var<storage, read_write> spatial_grid: array<atomic<u32>>; 

const GRID_DIM: u32 = 64u; // Must match Rust GRID_SIZE
const CELL_SIZE: f32 = 0.15; // Size of a "concept bucket"

// --- UTILS ---

fn hash_coords(pos: vec3<f32>) -> u32 {
    // Convert float position to grid index
    // Position range is roughly -3.0 to 3.0
    let grid_pos = vec3<u32>(clamp((pos + 3.0) / CELL_SIZE, vec3<f32>(0.0), vec3<f32>(f32(GRID_DIM) - 1.0)));
    return grid_pos.x + (grid_pos.y * GRID_DIM) + (grid_pos.z * GRID_DIM * GRID_DIM);
}

fn rand_vec3(seed: u32) -> vec3<f32> {
    let t = f32(seed) + params.time;
    return vec3<f32>(
        fract(sin(t * 12.9898) * 43758.5453),
        fract(sin(t * 78.233) * 43758.5453),
        fract(sin(t * 151.7182) * 43758.5453)
    ) * 2.0 - 1.0;
}

// --- PASS 1: CLEAR GRID ---
@compute @workgroup_size(64)
fn cs_clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&spatial_grid)) { return; }
    atomicStore(&spatial_grid[idx], 0xFFFFFFFFu); // Sentinel value
}

// --- PASS 2: POPULATE GRID (Semantic Layout) ---
@compute @workgroup_size(64)
fn cs_populate_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }
    
    let me = neurons[idx];
    let grid_idx = hash_coords(me.pos_semantic.xyz);
    
    // Write my ID into the grid. 
    // Race condition is intentional: we only need *some* neighbor in this cell.
    atomicStore(&spatial_grid[grid_idx], idx);
}

// --- PASS 3: UPDATE NEURONS (Sense, Dream, Learn) ---
@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }

    var me = neurons[idx];

    // 1. SENSORY INPUT (Reality)
    // --------------------------
    // Map physical position (screen space) to UV
    // pos_physical xy is in range [-aspect, aspect] roughly. Let's map -1..1 to 0..1
    let uv = me.pos_physical.xy * 0.5 + 0.5;
    let real_color = textureSampleLevel(input_texture, input_sampler, uv, 0.0);
    
    // Calculate brightness/activation from reality
    let input_activation = length(real_color.rgb);

    // 2. DREAM / PREDICTION (Inference)
    // ---------------------------------
    // Look at Semantic Neighbors to determine my next state.
    // "If my concept-neighbors are active, I should be active."
    
    var neighbor_activity_sum = 0.0;
    var cohesion_force = vec3<f32>(0.0);
    var neighbor_count = 0.0;

    let my_grid_idx = hash_coords(me.pos_semantic.xyz);

    // Check immediate neighborhood (3x3x3 grid)
    // Note: In a full implementation, we'd loop -1 to 1. 
    // For performance, we check 3 random offsets or just the center.
    // Let's check center + random offset.
    
    // Look at my own cell
    let neighbor_id = atomicLoad(&spatial_grid[my_grid_idx]);
    
    if (neighbor_id != 0xFFFFFFFFu && neighbor_id != idx) {
        let other = neurons[neighbor_id];
        
        // Semantic Distance
        let dist = distance(me.pos_semantic.xyz, other.pos_semantic.xyz);
        
        if (dist < 0.5) {
            // Standard Neural Summation
            neighbor_activity_sum += other.state.x; 
            
            // Swarming: Pull towards active neighbors
            if (other.state.x > 0.5) {
               cohesion_force += (other.pos_semantic.xyz - me.pos_semantic.xyz);
            }
            neighbor_count += 1.0;
        }
    }

    // My "Predicted" state is based on neighbors (Associative Memory)
    // If I have no neighbors, I predict 0.
    let predicted_activation = clamp(neighbor_activity_sum * 0.1, 0.0, 1.0);
    
    // 3. LEARNING (The Error Signal)
    // ------------------------------
    
    var error = 0.0;
    
    if (params.train_mode == 1u) {
        // Training Mode: Compare Prediction to Reality
        // Error = |My Prediction - What actually happened|
        error = abs(predicted_activation - input_activation);
        
        // Force my state to match reality (clamp to input) so the signal propagates
        me.state.x = mix(me.state.x, input_activation, 0.2);
    } else {
        // Dreaming Mode: My state is purely my prediction
        // The signal ripples through the semantic web without external correction
        me.state.x = mix(me.state.x, predicted_activation, 0.1);
        error = 0.0; // No stress in dreams
    }

    // 4. NEUROPLASTICITY (Movement)
    // -----------------------------
    
    let learning_rate = 0.02;
    
    if (error < 0.2 && params.train_mode == 1u && input_activation > 0.1) {
        // --- DOPAMINE (Low Error + High Activity) ---
        // "I correctly predicted this! I belong here."
        // Strengthen: Move closer to the neighbors that helped me predict.
        
        if (neighbor_count > 0.0) {
            me.velocity += vec4<f32>(cohesion_force * learning_rate * 2.0, 0.0);
        }
        
        // Reward: Decay velocity (crystallize)
        me.velocity *= 0.90;
        
    } else if (error > 0.4 && params.train_mode == 1u) {
        // --- TERROR (High Error) ---
        // "I was wrong! This concept position is invalid."
        // Panic: Jitter randomly to find a new semantic meaning.
        
        let panic_dir = rand_vec3(idx);
        me.velocity += vec4<f32>(panic_dir * learning_rate * 5.0, 0.0);
    }
    
    // 5. PHYSICS UPDATE
    // -----------------
    // Apply velocity
    me.pos_semantic += me.velocity * 0.1;
    // Drag/Friction
    me.velocity *= 0.95;
    // Centripetal force (keep them from flying off into infinity)
    me.velocity -= me.pos_semantic * 0.001;

    // 6. OUTPUT VISUALIZATION
    // -----------------------
    // Write to the output texture ("Visual Cortex")
    // We visualize the Neuron's PREDICTION, colored by its SEMANTIC POSITION.
    // This lets us see "Concepts" forming as colors.
    
    if (me.state.x > 0.1) {
        let dims = textureDimensions(output_texture);
        let out_uv = vec2<i32>(vec2<f32>(dims) * uv);
        
        // Color = Semantic Position (normalized) -> RGB
        // Brightness = Activation
        let concept_color = normalize(abs(me.pos_semantic.xyz)) * me.state.x;
        
        // We accumulate into the texture (since many neurons map to one pixel)
        // Note: Atomic float add isn't available, so we just overwrite or simple mix
        // For this demo, we just write.
        textureStore(output_texture, out_uv, vec4<f32>(concept_color, 1.0));
    }
    
    // Save state
    me.state.y = predicted_activation;
    me.state.z = error; // Trace
    neurons[idx] = me;
}

// --- RENDER SHADER (Vertex + Fragment) ---
// Visualizes the neurons as a 3D point cloud in the "Brain View"

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
    
    // Visualize Semantic Position (The "Mind Map")
    let world_pos = instance.pos_semantic.xyz + (model.position * 0.05);
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Color based on state:
    // White = Active
    // Red = High Error (Terror)
    // Blue = Low Error (Stable)
    let activity = instance.state.x;
    let error = instance.state.z;
    
    let base_color = normalize(abs(instance.pos_semantic.xyz));
    var final_color = mix(base_color * 0.2, vec3<f32>(1.0, 1.0, 1.0), activity);
    
    if (error > 0.3) {
        final_color = mix(final_color, vec3<f32>(1.0, 0.0, 0.0), 0.8); // Red flash on error
    }
    
    out.color = final_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}