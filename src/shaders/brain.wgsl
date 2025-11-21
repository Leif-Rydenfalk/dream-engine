// src/shaders/brain.wgsl

// --- Structs ---

struct Neuron {
    pos_physical: vec4<f32>, // xyz: Location in brain, w: unused
    pos_semantic: vec4<f32>, // xyz: Location in concept space, w: unused
    
    // State Vector:
    // x: Voltage (0.0 to 1.0)
    // y: Recovery/Fatigue (0.0 to 1.0)
    // z: Threshold (Homeostatic)
    // w: Trace (Recent firing memory for Hebbian learning)
    state: vec4<f32>, 
};

struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    mouse_x: f32,
    mouse_y: f32,
    is_clicking: u32,
    train_mode: u32, // 1 = Learning/Input, 0 = Dreaming
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    pos: vec3<f32>,
};

// --- Bindings ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var input_texture: texture_2d<f32>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba32float, write>;

// The "Field" - A Linearized 3D Density Map of semantic activity
// Replaces texture_storage_3d to allow generic atomics
@group(0) @binding(5) var<storage, read_write> concept_map: array<atomic<u32>>;

@group(1) @binding(0) var<uniform> camera: CameraUniform;

// --- Constants ---

const GRID_SIZE: u32 = 64u; // Must match Rust const
const GRID_SIZE_SQ: u32 = 4096u; // 64 * 64
const DT: f32 = 0.016;
const DECAY: f32 = 0.90;
const RECOVERY_RATE: f32 = 0.05;
const LEARNING_RATE: f32 = 0.01;
const NOISE: f32 = 0.001;

// --- Helpers ---

fn get_grid_index(p: vec3<i32>) -> u32 {
    return u32(p.x) + u32(p.y) * GRID_SIZE + u32(p.z) * GRID_SIZE_SQ;
}

// Hash function for pseudo-randomness
fn hash(n: u32) -> f32 {
    var x = n;
    x = (x << 13u) ^ x;
    return (1.0 - f32((x * (x * x * 15731u + 789221u) + 1376312589u) & 0x7fffffffu) / 1073741824.0);
}

fn float_to_uint(val: f32) -> u32 {
    return u32(max(0.0, val) * 1000.0);
}

fn uint_to_float(val: u32) -> f32 {
    return f32(val) / 1000.0;
}

// --- Compute Pass 1: Clear the Concept Field ---
@compute @workgroup_size(64)
fn cs_clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    // Total voxels = 64*64*64 = 262,144
    if (idx >= (GRID_SIZE * GRID_SIZE * GRID_SIZE)) { return; }
    atomicStore(&concept_map[idx], 0u);
}

// --- Compute Pass 2: Emit Signals & Handle Input ---
@compute @workgroup_size(64)
fn cs_populate_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }
    
    var n = neurons[idx];
    
    // 1. Sensory Input (Optic Nerve)
    if (params.train_mode == 1u && idx < (params.width * params.height)) {
        let uv = vec2<f32>(
            f32(idx % params.width) / f32(params.width),
            f32(idx / params.width) / f32(params.height)
        );
        let color = textureSampleLevel(input_texture, input_sampler, uv, 0.0);
        // Stimulation
        n.state.x += length(color.rgb) * 0.5; 
    }

    // 2. Emit to Field (Atomic Add to Buffer)
    if (n.state.x > 0.1) {
        let grid_pos = vec3<i32>((n.pos_semantic.xyz + 1.0) * 0.5 * f32(GRID_SIZE));
        
        if (grid_pos.x >= 0 && grid_pos.x < i32(GRID_SIZE) &&
            grid_pos.y >= 0 && grid_pos.y < i32(GRID_SIZE) &&
            grid_pos.z >= 0 && grid_pos.z < i32(GRID_SIZE)) {
            
            let density = float_to_uint(n.state.x);
            let buffer_idx = get_grid_index(grid_pos);
            atomicAdd(&concept_map[buffer_idx], density);
        }
    }

    neurons[idx] = n;
}

// --- Compute Pass 3: Update Logic & Plasticity ---
@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }
    
    var n = neurons[idx];
    var voltage = n.state.x;
    var recovery = n.state.y;
    var threshold = n.state.z;
    var trace = n.state.w;

    // --- 1. Synaptic Integration (Reading the Buffer) ---
    let grid_pos = vec3<i32>((n.pos_semantic.xyz + 1.0) * 0.5 * f32(GRID_SIZE));
    var input_current = 0.0;

    // Sample 3x3x3 volume
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let sample_pos = grid_pos + vec3<i32>(x, y, z);
                if (sample_pos.x >= 0 && sample_pos.x < i32(GRID_SIZE) &&
                    sample_pos.y >= 0 && sample_pos.y < i32(GRID_SIZE) &&
                    sample_pos.z >= 0 && sample_pos.z < i32(GRID_SIZE)) {
                    
                    let buffer_idx = get_grid_index(sample_pos);
                    let field_val = atomicLoad(&concept_map[buffer_idx]);
                    let density = uint_to_float(field_val);
                    
                    let dist = length(vec3<f32>(f32(x), f32(y), f32(z)));
                    input_current += density * (1.0 / (1.0 + dist * dist));
                }
            }
        }
    }

    input_current *= 0.05; 

    // --- 2. Dynamics ---
    voltage *= DECAY;
    voltage += input_current;
    
    if (recovery > 0.1) {
        voltage *= 0.5;
        recovery -= RECOVERY_RATE;
    }

    if (params.train_mode == 0u) {
         voltage += (hash(idx + u32(params.time * 1000.0)) - 0.5) * 0.05;
    }

    var fired = false;
    if (voltage > threshold) {
        fired = true;
        voltage = -0.1;
        recovery = 1.0;
        trace = 1.0;
        threshold += 0.002;
    } else {
        threshold = max(0.1, threshold - 0.00001);
        trace *= 0.95;
    }

    // --- 3. Plasticity ---
    if (params.train_mode == 1u) {
        if (trace > 0.5) {
             let rnd = vec3<f32>(
                hash(idx * 3u) - 0.5, 
                hash(idx * 3u + 1u) - 0.5, 
                hash(idx * 3u + 2u) - 0.5
             );
             n.pos_semantic = vec4<f32>(normalize(n.pos_semantic.xyz + rnd * LEARNING_RATE), 1.0);
        }
    }

    // --- 4. Output ---
    if (idx < (params.width * params.height)) {
        let out_uv = vec2<i32>(
            i32(idx) % i32(params.width),
            i32(idx) / i32(params.width)
        );
        let color = vec4<f32>(n.pos_semantic.xyz * 0.5 + 0.5, 1.0) * (voltage + trace);
        textureStore(output_texture, out_uv, color);
    }

    n.state = vec4<f32>(voltage, recovery, threshold, trace);
    neurons[idx] = n;
}

// --- Rendering (Vertex & Fragment) ---

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
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

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    let scale = 0.02 + (instance.state.x * 0.05); 
    let world_pos = instance.pos_physical.xyz + (model.position * scale);
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    let semantic_color = instance.pos_semantic.xyz * 0.5 + 0.5;
    let active_color = vec3<f32>(1.0, 0.2, 0.1);
    
    out.color = mix(semantic_color * 0.2, active_color, instance.state.x);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}