// --- STRUCTS ---

struct Neuron {
    // Block 1 (16 bytes)
    concept_pos: vec4<f32>,
    
    // Block 2 (16 bytes)
    cortical_pos: vec2<f32>,
    retinal_coord: vec2<u32>,
    
    // Block 3 (16 bytes)
    layer: u32,
    voltage: f32,
    plasticity_trace: f32,
    learning_rate: f32,

    // Block 4 (16 bytes)
    surprise_accumulator: f32,
    top_geometric_contrib: f32,
    top_geometric_source: u32,
    pad0: f32,

    // Explicit Synapses (Fixed 16 slots for cache alignment)
    // 16 * 4 bytes = 64 bytes each array
    explicit_targets: array<u32, 16>,
    explicit_weights: array<f32, 16>,
    explicit_ages: array<f32, 16>,
    explicit_visual_weights: array<f32, 16>,
};

struct SynapticKernel {
    local_amplitude: f32,
    local_decay: f32,
    conceptual_amplitude: f32,
    conceptual_decay: f32,
    inhibit_radius: f32,
    inhibit_strength: f32,
    explicit_learning_rate: f32,
    pruning_threshold: f32,
    promotion_threshold: f32,
    temporal_decay: f32,
    pad: vec2<f32>,
};

struct SimParams {
    neuron_count: u32,
    time: f32,
    dt: f32,
    geometric_sample_count: u32,
    explicit_synapse_slots: u32,
    train_mode: u32, 
    terror_threshold: f32,
    grid_dim: u32,
    use_camera: u32, 
    pad: vec3<f32>, 
};

struct LineVertex {
    pos: vec4<f32>,
    color: vec4<f32>,
};

// --- BINDINGS ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<uniform> kernel: SynapticKernel;
@group(0) @binding(3) var<storage, read_write> spatial_grid: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> spike_history: array<f32>;
@group(0) @binding(5) var<storage, read_write> line_buffer: array<LineVertex>;

@group(0) @binding(6) var input_tex: texture_2d<f32>;
@group(0) @binding(7) var input_sampler: sampler;
@group(0) @binding(8) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var prediction_tex: texture_storage_2d<rgba32float, write>;

// --- UTILITY ---

fn hash_to_grid(pos: vec3<f32>) -> u32 {
    let grid_size = f32(params.grid_dim);
    let scaled = (pos + vec3<f32>(2.0)) / 4.0;
    let cell = vec3<u32>(clamp(scaled * grid_size, vec3<f32>(0.0), vec3<f32>(grid_size - 1.0)));
    return cell.x + cell.y * params.grid_dim + cell.z * params.grid_dim * params.grid_dim;
}

fn rand(seed: u32) -> f32 {
    return fract(sin(f32(seed) * 12.9898) * 43758.5453);
}

fn rand_vec3_stable(seed: u32, time: f32) -> vec3<f32> {
    let t = floor(time * 30.0);
    let s = f32(seed) + t;
    return vec3<f32>(
        fract(sin(s * 12.9898) * 43758.5453),
        fract(sin(s * 78.233) * 43758.5453),
        fract(sin(s * 44.123) * 43758.5453)
    ) * 2.0 - 1.0;
}

// --- COMPUTE SHADERS ---

@compute @workgroup_size(64)
fn cs_init_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    var n: Neuron;
    let seed = idx * 1000u;
    
    // Reduced Retina Dim for optimization (160x160 = 25,600 neurons)
    let retina_dim = 160u;
    let retina_count = retina_dim * retina_dim; 
    
    if (idx < retina_count) {
        // RETINA LAYER (Layer 0)
        n.layer = 0u;
        n.retinal_coord = vec2<u32>(idx % retina_dim, idx / retina_dim);
        let u = f32(n.retinal_coord.x) / f32(retina_dim);
        let v = f32(n.retinal_coord.y) / f32(retina_dim);
        n.cortical_pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.concept_pos = vec4<f32>(n.cortical_pos, -0.8, 0.0); 
        n.learning_rate = 0.0;
    } else {
        // CORTEX LAYER (Layer 1)
        n.layer = 1u;
        n.retinal_coord = vec2<u32>(0u, 0u);
        n.cortical_pos = vec2<f32>(rand(seed), rand(seed+1u)) * 2.0 - 1.0;
        n.concept_pos = vec4<f32>(n.cortical_pos, 0.0 + (rand(seed+2u)*0.4 - 0.2), 0.0);
        n.learning_rate = 0.1;
    }

    // Reduced explicit slots to 16
    for (var i = 0; i < 16; i++) {
        n.explicit_targets[i] = 0xFFFFFFFFu;
        n.explicit_weights[i] = 0.0;
        n.explicit_visual_weights[i] = 0.0;
        n.explicit_ages[i] = 0.0;
    }
    
    n.voltage = 0.0;
    n.plasticity_trace = 0.0;
    n.surprise_accumulator = 0.0;
    n.top_geometric_contrib = 0.0;
    n.top_geometric_source = 0xFFFFFFFFu;
    
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
    let n = neurons[idx];
    
    let concept_hash = hash_to_grid(n.concept_pos.xyz);
    atomicStore(&spatial_grid[concept_hash], idx);
    
    if (n.layer == 0u) {
        let fuzzy_hash = hash_to_grid(n.concept_pos.xyz + vec3<f32>(0.01));
        atomicStore(&spatial_grid[fuzzy_hash], idx);
    }
}

@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let post_idx = id.x;
    if (post_idx >= params.neuron_count) { return; }
    
    var post = neurons[post_idx];
    let retina_dim = 160u; // Matches Init

    // ========================================
    // LAYER 0: RETINA
    // ========================================
    if (post.layer == 0u) {
        let uv = vec2<f32>(post.retinal_coord) / f32(retina_dim);
        
        var reality = 0.0;
        if (params.use_camera == 1u) {
            reality = textureSampleLevel(input_tex, input_sampler, uv, 0.0).r;
        } else {
            let zoom_out = 1.02; 
            let drift = vec2<f32>(sin(params.time * 0.5), cos(params.time * 0.4)) * 0.02;
            let dream_uv = (uv - 0.5) * zoom_out + 0.5 + drift;
            reality = textureSampleLevel(input_tex, input_sampler, dream_uv, 0.0).r;
            reality += (rand(post_idx + u32(params.time * 100.0)) - 0.5) * 0.05;
        }
        
        let reality_val = (reality - 0.5) * 3.0; 

        // Simplified prediction sampling (fewer samples for perf)
        var prediction_sum = 0.0;
        var prediction_count = 0.0;
        
        // Reduced from 12 to 4 samples
        for (var i = 0u; i < 4u; i++) {
            let offset = rand_vec3_stable(post_idx * 50u + i, params.time) * 0.15;
            let probe_pos = vec3<f32>(post.cortical_pos, 0.0) + offset;
            let grid_idx = hash_to_grid(probe_pos);
            let pre_idx = atomicLoad(&spatial_grid[grid_idx]);
            
            if (pre_idx != 0xFFFFFFFFu && pre_idx < params.neuron_count) {
                let pre = neurons[pre_idx];
                if (pre.layer == 1u) {
                    // Manually compute distance squared to avoid sqrt
                    let d = vec3<f32>(post.cortical_pos, 0.0) - pre.concept_pos.xyz;
                    let dist_sq = dot(d, d);
                    let weight = exp(-dist_sq * 20.0);
                    prediction_sum += pre.plasticity_trace * weight;
                    prediction_count += weight;
                }
            }
        }
        
        let prediction = select(0.0, prediction_sum / prediction_count, prediction_count > 0.001);
        let error = reality_val - prediction;
        
        post.plasticity_trace = reality_val;
        post.voltage = error * 5.0;
        post.surprise_accumulator = mix(post.surprise_accumulator, abs(error), 0.1);
        
        neurons[post_idx] = post;
        return;
    }

    // ========================================
    // LAYER 1: CORTEX
    // ========================================
    var input_signal = 0.0;
    var input_error = 0.0;

    // OPTIMIZATION: Hardcoded 16 loop for compiler unrolling
    for (var i = 0; i < 16; i++) {
        let pre_idx = post.explicit_targets[i];
        if (pre_idx == 0xFFFFFFFFu) { continue; }
        
        let pre = neurons[pre_idx];
        let w = post.explicit_weights[i];
        
        input_signal += w * pre.plasticity_trace;
        post.explicit_ages[i] += params.dt;
        
        let target_vis = select(0.0, abs(w), abs(w) > 0.1);
        post.explicit_visual_weights[i] += (target_vis - post.explicit_visual_weights[i]) * 0.1;
    }

    // OPTIMIZATION: Structured Tiled Sampling (No random memory thrashing)
    var best_activation = 0.0;
    var best_source = 0xFFFFFFFFu;
    
    // 4x4 Grid Sample = 16 reads (down from 64 random reads)
    for (var i = 0u; i < 16u; i++) {
        let tile_x = i % 4u;
        let tile_y = i / 4u;
        let offset = vec2<f32>(f32(tile_x) - 1.5, f32(tile_y) - 1.5) * 0.05; // Tiling
        
        let sample_pos_2d = post.cortical_pos + offset;
        
        let retina_u = (sample_pos_2d.x + 1.0) * 0.5;
        let retina_v = (sample_pos_2d.y + 1.0) * 0.5;
        
        if (retina_u >= 0.0 && retina_u < 1.0 && retina_v >= 0.0 && retina_v < 1.0) {
            let rx = u32(retina_u * 160.0); // 160.0 match retina_dim
            let ry = u32(retina_v * 160.0);
            let pre_idx = rx + ry * 160u;
            
            if (pre_idx < 25600u) { // 160*160
                let pre = neurons[pre_idx];
                
                let d = sample_pos_2d - pre.cortical_pos;
                let dist_sq = dot(d, d);
                let weight = exp(-dist_sq * 20.0);
                
                let visual_input = pre.plasticity_trace * weight;
                let error_signal = pre.voltage * weight;
                
                input_signal += visual_input;
                input_error += error_signal;
                
                let activation = abs(visual_input) * weight;
                if (activation > best_activation) {
                    best_activation = activation;
                    best_source = pre_idx;
                }
            }
        }
    }
    
    post.top_geometric_contrib = best_activation;
    post.top_geometric_source = best_source;

    post.voltage *= 0.90;
    post.voltage += (input_signal + input_error * 0.5) * params.dt * 10.0;
    post.voltage = clamp(post.voltage, -3.0, 3.0);
    post.plasticity_trace += (post.voltage - post.plasticity_trace) * 0.2;
    
    // ========================================
    // LEARNING
    // ========================================
    if (params.train_mode == 1u) {
        let learning_rate = post.learning_rate * kernel.explicit_learning_rate;
        
        for (var i = 0; i < 16; i++) {
            let pre_idx = post.explicit_targets[i];
            if (pre_idx == 0xFFFFFFFFu) { continue; }
            let pre = neurons[pre_idx];
            
            let delta = post.voltage * pre.plasticity_trace; 
            post.explicit_weights[i] += delta * learning_rate;
            post.explicit_weights[i] *= 0.9995;
            
            if (post.explicit_ages[i] > kernel.pruning_threshold) {
                 if (abs(post.explicit_weights[i]) < 0.05) {
                     post.explicit_targets[i] = 0xFFFFFFFFu;
                     post.explicit_weights[i] = 0.0;
                 }
            }
        }
        
        if (best_activation > kernel.promotion_threshold && best_source != 0xFFFFFFFFu) {
            var weakest_idx = 0;
            var weakest_strength = abs(post.explicit_weights[0]);
            var found_empty = false;
            
            for (var i = 0; i < 16; i++) {
                if (post.explicit_targets[i] == 0xFFFFFFFFu) {
                    weakest_idx = i;
                    found_empty = true;
                    break;
                }
                let strength = abs(post.explicit_weights[i]);
                if (strength < weakest_strength) {
                    weakest_strength = strength;
                    weakest_idx = i;
                }
            }
            
            if (found_empty || best_activation > weakest_strength * 2.0) {
                post.explicit_targets[weakest_idx] = best_source;
                post.explicit_weights[weakest_idx] = best_activation * 0.5;
                post.explicit_ages[weakest_idx] = 0.0;
            }
        }
    }
    
    post.surprise_accumulator = mix(post.surprise_accumulator, abs(input_signal + input_error), 0.05);
    neurons[post_idx] = post;
    
    // Write to texture (1 pixel)
    let cortex_uv = (post.cortical_pos + 1.0) * 0.5;
    let tex_x = i32(cortex_uv.x * 512.0);
    let tex_y = i32(cortex_uv.y * 512.0);
    
    if (tex_x >= 0 && tex_x < 512 && tex_y >= 0 && tex_y < 512) {
        let activity = post.voltage;
        let hot = vec3<f32>(1.0, 0.5, 0.0) * max(0.0, activity * 0.5);
        let cold = vec3<f32>(0.0, 0.3, 1.0) * max(0.0, -activity * 0.5);
        let col = vec4<f32>(hot + cold, 1.0);
        textureStore(output_tex, vec2<i32>(tex_x, tex_y), col);
    }
}

@compute @workgroup_size(64)
fn cs_generate_lines(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    let neuron = neurons[idx];
    let start_offset = idx * 16u * 2u; // 16 slots
    let origin = neuron.concept_pos.xyz * 3.0; 
    
    for (var i = 0u; i < 16u; i++) {
        let target_idx = neuron.explicit_targets[i];
        let vis_weight = neuron.explicit_visual_weights[i]; 
        let buffer_idx = start_offset + i * 2u;
        
        if (target_idx != 0xFFFFFFFFu && target_idx < params.neuron_count && vis_weight > 0.01) {
            let target_neuron = neurons[target_idx];
            let dest = target_neuron.concept_pos.xyz * 3.0;
            let weight = neuron.explicit_weights[i];
            
            var col = select(vec3<f32>(1.0, 0.1, 0.1), vec3<f32>(0.0, 0.9, 1.0), weight > 0.0);
            let alpha = clamp(vis_weight * 2.0, 0.0, 0.6);

            line_buffer[buffer_idx].pos = vec4<f32>(origin, 1.0);
            line_buffer[buffer_idx].color = vec4<f32>(col, alpha);
            line_buffer[buffer_idx+1].pos = vec4<f32>(dest, 1.0);
            line_buffer[buffer_idx+1].color = vec4<f32>(col, alpha);
        } else {
            line_buffer[buffer_idx].pos = vec4<f32>(0.0);
            line_buffer[buffer_idx].color = vec4<f32>(0.0);
            line_buffer[buffer_idx+1].pos = vec4<f32>(0.0);
            line_buffer[buffer_idx+1].color = vec4<f32>(0.0);
        }
    }
}

// --- VISUALIZERS ---

@compute @workgroup_size(8, 8)
fn cs_render_dream(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    let uv = vec2<f32>(id.xy) / vec2<f32>(dim);
    // Matches new retina dim 160
    let retinal_x = u32(uv.x * 160.0);
    let retinal_y = u32(uv.y * 160.0);
    let idx = retinal_x + retinal_y * 160u;
    
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    if (idx < params.neuron_count) {
        let n = neurons[idx];
        let pred = n.plasticity_trace * 0.5 + 0.5; 
        color = vec4<f32>(pred, pred, pred, 1.0);
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec3<f32>,
    padding: u32,
    time: f32,
}

@group(1) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(
    @location(0) vertex_pos: vec3<f32>,
    @builtin(instance_index) instance_idx: u32
) -> VertexOutput {
    var out: VertexOutput;
    
    let neuron = neurons[instance_idx];
    let world_pos = neuron.concept_pos.xyz * 3.0 + vertex_pos * 0.015; 
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    var base_color = vec3<f32>(0.1);
    
    if (neuron.layer == 0u) { 
        let err = neuron.voltage * 0.1; 
        base_color = vec3<f32>(0.1) + vec3<f32>(max(0.0, err), 0.0, max(0.0, -err));
    } 
    else if (neuron.layer == 1u) { 
        let act = neuron.voltage * 0.3;
        base_color = vec3<f32>(max(0.0, act), max(0.0, act*0.5), abs(neuron.plasticity_trace)*0.3);
    }

    out.color = vec4<f32>(base_color, 1.0);
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}

@vertex
fn vs_lines(
    @location(0) pos: vec4<f32>, 
    @location(1) col: vec4<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos.xyz, 1.0);
    out.color = col;
    return out;
}

@fragment
fn fs_lines(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}