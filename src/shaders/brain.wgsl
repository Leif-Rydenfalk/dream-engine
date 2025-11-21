// src/shaders/brain.wgsl

struct Neuron {
    pos: vec4f,        // xyz: Tissue Position, w: padding
    weight: vec4f,     // xyz: RGB (Concept), w: padding
    state: vec4f,      // x: Potential (Voltage), y: Recovery, z: Firing? (1.0/0.0), w: padding
}

struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    mouse_x: f32,
    mouse_y: f32,
    is_clicking: u32,
}

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var input_texture: texture_2d<f32>; // Camera Feed
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba32float, write>; // Hallucination

// Pseudo-random number generator
fn hash(n: u32) -> f32 {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return f32(x) / 4294967295.0;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) { return; }

    var n = neurons[i];
    
    // --- 1. SENSORY INPUT (The Eyes) ---
    // The first % of neurons are mapped to the retina (screen)
    let retina_count = params.count / 4u; // 25% are sensory neurons
    var external_input = 0.0;
    
    if (i < retina_count) {
        // Map 1D index to 2D UV
        let row_size = u32(sqrt(f32(retina_count)));
        let uv = vec2f(f32(i % row_size), f32(i / row_size)) / f32(row_size);
        
        let color = textureSampleLevel(input_texture, input_sampler, uv, 0.0).rgb;
        
        // If the neuron's "concept" (weight) matches the pixel color, it gets excited
        let match_score = 1.0 - distance(n.weight.rgb, color);
        external_input += max(0.0, match_score) * 0.05;
        
        // Force weight alignment for sensory neurons (Retinal Imprinting)
        n.weight = mix(n.weight, vec4f(color, 1.0), 0.1); 
    }

    // --- 2. INTERACTION (The Mouse) ---
    // Inject current if mouse is close to Tissue Position
    let aspect = f32(params.width) / f32(params.height);
    // Project tissue pos to roughly -1 to 1 range for 2D comparison
    let screen_pos = n.pos.xy; 
    let mouse_dist = distance(screen_pos, vec2f(params.mouse_x, params.mouse_y));
    
    if (params.is_clicking != 0u && mouse_dist < 0.2) {
        external_input += 0.5;
    }

    // --- 3. DYNAMICS (Izhikevich / Integrate-and-Fire approximation) ---
    // Randomly sample neighbors to simulate connectivity
    let seed = i + u32(params.time * 60.0);
    var lateral_input = 0.0;
    
    // Stochastic Connectivity: Sample 16 random partners
    for (var k = 0u; k < 16u; k++) {
        let neighbor_idx = u32(hash(seed + k) * f32(params.count));
        let neighbor = neurons[neighbor_idx];
        
        // If neighbor is firing
        if (neighbor.state.z > 0.5) {
            // Check SEMANTIC distance (Weight Space)
            let semantic_dist = distance(n.weight.xyz, neighbor.weight.xyz);
            
            // Connect if concepts are similar ("Red" neurons talk to "Orange" neurons)
            if (semantic_dist < 0.2) {
                lateral_input += 0.01;
                
                // --- 4. LEARNING (Hebbian) ---
                // If I am also excited, move my weight closer to this firing neighbor
                if (n.state.x > 0.5) {
                    n.weight = mix(n.weight, neighbor.weight, 0.005);
                }
            }
        }
    }

    // Update Potential
    // dx/dt = Input - Decay
    n.state.x += external_input + lateral_input - 0.01;
    
    // Fire?
    n.state.z = 0.0; // Reset fire flag
    if (n.state.x > 1.0) {
        n.state.z = 1.0; // SPIKE!
        n.state.x = -0.5; // Refractory period
        
        // Flash the weight color slightly towards white when firing (visual effect)
        // n.weight.x = min(n.weight.x + 0.1, 1.0);
    }
    
    // Decay back to resting potential
    n.state.x = mix(n.state.x, 0.0, 0.05);

    neurons[i] = n;

    // --- 5. HALLUCINATION (Output) ---
    // Write to the output texture if this neuron is firing
    // We map the neuron's "Concept" (Weight Color) to its "Location" (Tissue Pos)
    if (n.state.z > 0.5) {
        // Map tissue pos (-1 to 1) to texture coords (0 to params.width)
        // Simple projection: Use X/Y of tissue
        // Normalize roughly -2..2 to 0..1
        let u = (n.pos.x * 0.5) + 0.5;
        let v = 1.0 - ((n.pos.y * 0.5) + 0.5); // Flip Y
        
        let tx = i32(u * f32(params.width));
        let ty = i32(v * f32(params.height));
        
        if (tx >= 0 && tx < i32(params.width) && ty >= 0 && ty < i32(params.height)) {
            // Write the concept color to the texture
            // We use atomic-like behavior by relying on persistence or just overwriting
            textureStore(output_texture, vec2<i32>(tx, ty), vec4f(n.weight.rgb, 1.0));
        }
    }
}


struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    pos: vec3f,
    time: f32,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_uv: vec2f,
    @location(2) normal: vec3f,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec3f,
    @location(1) normal: vec3f,
};

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) idx: u32) -> VertexOutput {
    let n = neurons[idx];
    
    // Use Tissue Position
    let scale = 0.03;
    let world_pos = in.position * scale + n.pos.xyz;
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4f(world_pos, 1.0);
    
    // Visualize State:
    // Base color = Weight (Concept)
    // Add brightness if firing (State.z)
    let firing = n.state.z;
    out.color = n.weight.rgb + vec3f(firing * 2.0); // Flash white on fire
    out.normal = in.normal;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let light = normalize(vec3f(0.5, 1.0, 0.5));
    let diff = max(dot(in.normal, light), 0.2);
    return vec4f(in.color * diff, 1.0);
}