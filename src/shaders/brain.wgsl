// --- Structures matching Rust ---

struct Neuron {
    pos: vec4<f32>,    // xyz: Position, w: unused
    weight: vec4<f32>, // xyz: Semantic Weight, w: Strength
    state: vec4<f32>,  // x: Voltage, y: Fatigue, z: Firing (0/1), w: Unused
}

struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    mouse_x: f32,
    mouse_y: f32,
    is_clicking: u32,
    train_mode: u32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    pos: vec3<f32>,
    padding: u32,
    time: f32,
}

// --- Bindings ---
// Group 0: Brain System
@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var input_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var concept_map: texture_storage_3d<r32uint, read_write>;

// Group 1: Camera (For rendering 3D view)
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// --- Constants ---
const GRID_SIZE: u32 = 32u;

// --- COMPUTE: Clear Grid ---
// Clears the 3D spatial map before neurons write to it
@compute @workgroup_size(8, 8, 8)
fn cs_clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= GRID_SIZE || id.y >= GRID_SIZE || id.z >= GRID_SIZE) {
        return;
    }
    // FIX: Cast id to vec3<i32> for textureStore
    textureStore(concept_map, vec3<i32>(id), vec4<u32>(0u));
}

// --- COMPUTE: Populate Grid ---
// Neurons register their index into the 3D voxel grid
@compute @workgroup_size(64, 1, 1)
fn cs_populate_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }

    let pos = neurons[idx].pos.xyz;
    
    // Map world position (-2.5 to 2.5) to Grid Coords (0 to 32)
    let grid_pos = vec3<u32>((pos + vec3<f32>(2.5)) * (f32(GRID_SIZE) / 5.0));
    
    if (grid_pos.x < GRID_SIZE && grid_pos.y < GRID_SIZE && grid_pos.z < GRID_SIZE) {
        // FIX: Cast grid_pos to vec3<i32>
        textureStore(concept_map, vec3<i32>(grid_pos), vec4<u32>(idx));
    }
}

// --- COMPUTE: Update Neurons ---
// The main brain loop: Read Input -> Calculate Voltage -> Fire -> Write Output
@compute @workgroup_size(64, 1, 1)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) { return; }

    var neuron = neurons[idx];

    // 1. Read Sensory Input (Optic Nerve)
    // Map 3D position to 2D UV coordinates (Projecting Z depth to flat image)
    let uv = (neuron.pos.xy / 5.0) + 0.5; // Map -2.5..2.5 to 0..1
    
    var sensory_input: vec4<f32> = vec4<f32>(0.0);
    
    if (params.train_mode == 1u && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
        sensory_input = textureSampleLevel(input_tex, samp, uv, 0.0);
    }

    // 2. Implicit Connectivity (Dendrites)
    let grid_pos = vec3<u32>((neuron.pos.xyz + vec3<f32>(2.5)) * (f32(GRID_SIZE) / 5.0));
    var neighbor_input: f32 = 0.0;
    
    // FIX: Removed the 3rd argument (0) and cast grid_pos to i32
    let neighbor_id = textureLoad(concept_map, vec3<i32>(grid_pos)).r;
    
    if (neighbor_id != 0u && neighbor_id != idx) {
        let neighbor = neurons[neighbor_id];
        if (neighbor.state.z > 0.5) { // If neighbor is firing
            neighbor_input += 0.05; // Excite
        }
    }

    // 3. Calculate Voltage
    let perception_match = dot(sensory_input.rgb, neuron.weight.rgb);
    var delta_voltage = (perception_match * 0.1) + neighbor_input;
    
    // Apply mouse stimulus
    let mouse_dist = distance(vec2<f32>(params.mouse_x, params.mouse_y), neuron.pos.xy / 2.5);
    if (params.is_clicking == 1u && mouse_dist < 0.2) {
        delta_voltage += 0.5;
    }

    // Apply to state
    neuron.state.x += delta_voltage;
    neuron.state.x *= 0.95; // Decay

    // 4. Fire?
    if (neuron.state.x > 0.8 && neuron.state.y < 0.1) {
        neuron.state.z = 1.0; // FIRE
        neuron.state.y = 1.0; // Fatigue
        neuron.state.x = -0.2; // Hyperpolarization
    } else {
        neuron.state.z = 0.0; 
        neuron.state.y *= 0.98;
    }

    // 5. Hallucinate
    if (neuron.state.z > 0.5) {
        let pixel_coord = vec2<i32>(uv * 512.0);
        if (pixel_coord.x >= 0 && pixel_coord.x < 512 && pixel_coord.y >= 0 && pixel_coord.y < 512) {
            let output_color = vec4<f32>(neuron.weight.rgb, 1.0);
            textureStore(output_tex, pixel_coord, output_color);
        }
    } else {
        // Fade trail
        if (params.time % 0.1 < 0.01) {
             let pixel_coord = vec2<i32>(uv * 512.0);
             if (pixel_coord.x >= 0 && pixel_coord.x < 512 && pixel_coord.y >= 0 && pixel_coord.y < 512) {
                 textureStore(output_tex, pixel_coord, vec4<f32>(0.0));
             }
        }
    }

    neurons[idx] = neuron;
}

// --- RENDER PIPELINE ---

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_idx: u32,
    model: VertexInput,
) -> VertexOutput {
    let neuron = neurons[instance_idx];
    
    let scale = 0.03;
    var world_pos = model.position * scale + neuron.pos.xyz;

    var color = neuron.weight.rgb;
    if (neuron.state.z > 0.5) {
        color = vec3<f32>(1.0, 1.0, 1.0);
        world_pos = model.position * (scale * 2.0) + neuron.pos.xyz;
    } else {
        color = mix(color, vec3<f32>(0.0), neuron.state.y);
    }

    var out: VertexOutput;
    out.uv = model.uv;
    out.color = color;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}