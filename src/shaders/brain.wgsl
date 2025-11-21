struct Neuron {
    // GEOMETRIC (0-48)
    concept_pos: vec3<f32>,          
    pad1: f32,                       
    cortical_pos: vec2<f32>,         
    pad2: vec2<f32>,                 
    receptive_center: vec3<f32>,     
    receptive_scale: f32,            
    
    // EXPLICIT (48-432)
    explicit_targets: array<u32, 32>, 
    explicit_weights: array<f32, 32>,   
    explicit_ages: array<f32, 32>,      
    
    // PLASTICITY (432-468)
    learning_rate: f32,
    homeostatic_target: f32,
    plasticity_trace: f32,           
    surprise_accumulator: f32,        
    voltage: f32,
    spike_time: f32,
    refractory_period: f32,
    top_geometric_contrib: f32,      
    top_geometric_source: u32,       
    
    // PADDING (468-480)
    padding_final: vec3<f32>,
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
    pad: vec4<f32>,
};

struct LineVertex {
    pos: vec3<f32>,
    pad: f32,
    color: vec3<f32>,
    pad2: f32,
};

// --- BINDINGS ---

@group(0) @binding(0) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<uniform> kernel: SynapticKernel;
@group(0) @binding(3) var<storage, read_write> spatial_grid: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> spike_history: array<f32>;
@group(0) @binding(5) var<storage, read_write> line_buffer: array<LineVertex>;

// --- UTILITY ---

fn hash_to_grid(pos: vec3<f32>) -> u32 {
    let grid_size = f32(params.grid_dim);
    let scaled = (pos + vec3<f32>(4.0)) / 8.0;
    let cell = vec3<u32>(clamp(scaled * grid_size, vec3<f32>(0.0), vec3<f32>(grid_size - 1.0)));
    return cell.x + cell.y * params.grid_dim + cell.z * params.grid_dim * params.grid_dim;
}

fn rand(seed: u32) -> f32 {
    let t = f32(seed) + params.time * 1000.0;
    return fract(sin(t * 12.9898) * 43758.5453);
}

fn rand_vec3(seed: u32) -> vec3<f32> {
    return vec3<f32>(rand(seed), rand(seed + 1000u), rand(seed + 2000u)) * 2.0 - 1.0;
}

fn compute_geometric_weight(pre: Neuron, post: Neuron) -> f32 {
    var weight = 0.0;
    let cortical_dist = distance(pre.cortical_pos, post.cortical_pos);
    weight += kernel.local_amplitude * exp(-cortical_dist * cortical_dist / kernel.local_decay);
    
    let concept_dist = distance(pre.concept_pos, post.concept_pos);
    weight += kernel.conceptual_amplitude * exp(-concept_dist * concept_dist / kernel.conceptual_decay);
    
    if (cortical_dist < kernel.inhibit_radius) {
        weight -= kernel.inhibit_strength * (1.0 - cortical_dist / kernel.inhibit_radius);
    }
    weight *= (1.0 + post.plasticity_trace * 0.2);
    return weight;
}

// --- EXPLICIT SYNAPSE HELPERS ---

fn find_explicit_synapse(post: ptr<function, Neuron>, pre_idx: u32) -> i32 {
    for (var i = 0; i < 32; i++) {
        if ((*post).explicit_targets[i] == pre_idx) { return i; }
    }
    return -1;
}

fn find_empty_slot(post: ptr<function, Neuron>) -> i32 {
    for (var i = 0; i < 32; i++) {
        if ((*post).explicit_targets[i] == 0xFFFFFFFFu) { return i; }
    }
    return -1;
}

fn find_weakest_synapse(post: ptr<function, Neuron>) -> i32 {
    var weakest_idx = 0;
    var weakest_strength = abs((*post).explicit_weights[0]);
    for (var i = 1; i < 32; i++) {
        let strength = abs((*post).explicit_weights[i]);
        if (strength < weakest_strength) {
            weakest_strength = strength;
            weakest_idx = i;
        }
    }
    return weakest_idx;
}

fn promote_to_explicit(post: ptr<function, Neuron>, pre_idx: u32, initial_weight: f32) {
    var slot = find_empty_slot(post);
    if (slot < 0) { slot = find_weakest_synapse(post); }
    if (slot >= 0) {
        (*post).explicit_targets[slot] = pre_idx;
        (*post).explicit_weights[slot] = initial_weight;
        (*post).explicit_ages[slot] = 0.0;
    }
}

fn prune_explicit_synapses(post: ptr<function, Neuron>) {
    for (var i = 0; i < 32; i++) {
        (*post).explicit_ages[i] += params.dt;
        if ((*post).explicit_ages[i] > kernel.pruning_threshold) {
            (*post).explicit_targets[i] = 0xFFFFFFFFu;
            (*post).explicit_weights[i] = 0.0;
            (*post).explicit_ages[i] = 0.0;
        }
    }
}

// --- COMPUTE SHADERS ---

@compute @workgroup_size(64)
fn cs_init_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    var n: Neuron;
    let seed = idx * 1000u;
    n.concept_pos = rand_vec3(seed) * 2.0;
    n.cortical_pos = vec2<f32>(rand(seed + 100u), rand(seed + 101u)) * 2.0 - 1.0;
    n.receptive_center = rand_vec3(seed + 200u);
    n.receptive_scale = 0.5 + rand(seed + 300u) * 0.5;
    
    for (var i = 0; i < 32; i++) {
        n.explicit_targets[i] = 0xFFFFFFFFu;
        n.explicit_weights[i] = 0.0;
        n.explicit_ages[i] = 0.0;
    }
    
    n.learning_rate = 0.001;
    n.homeostatic_target = 0.2;
    n.voltage = rand(seed + 400u) * 0.1;
    n.refractory_period = 0.02;
    
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
    
    let concept_hash = hash_to_grid(n.concept_pos);
    atomicStore(&spatial_grid[concept_hash], idx);
    
    let cortical_hash = hash_to_grid(vec3<f32>(n.cortical_pos, 0.0));
    atomicStore(&spatial_grid[cortical_hash], idx);
}

@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let post_idx = id.x;
    if (post_idx >= params.neuron_count) { return; }
    
    var post = neurons[post_idx];
    
    var geometric_input = 0.0;
    var explicit_input = 0.0;
    var max_geometric_contrib = 0.0;
    var max_geometric_source = 0xFFFFFFFFu;
    
    // Explicit Synapses
    for (var i = 0; i < 32; i++) {
        let pre_idx = post.explicit_targets[i];
        if (pre_idx == 0xFFFFFFFFu || pre_idx >= params.neuron_count) { continue; }
        
        let pre = neurons[pre_idx];
        let weight = post.explicit_weights[i];
        let pre_activity = pre.voltage;
        
        let spike_time_diff = params.time - pre.spike_time;
        let temporal_factor = exp(-spike_time_diff / kernel.temporal_decay);
        explicit_input += weight * pre_activity * temporal_factor;
    }
    
    // Geometric Sampling
    let local_samples = params.geometric_sample_count / 2u;
    var geometric_sample_count = 0.0;
    
    for (var i = 0u; i < local_samples; i++) {
        let offset = rand_vec3(post_idx * 1000u + i) * 0.2;
        let probe_pos = vec3<f32>(post.cortical_pos, 0.0) + offset;
        let grid_idx = hash_to_grid(probe_pos);
        let pre_idx = atomicLoad(&spatial_grid[grid_idx]);
        
        if (pre_idx != 0xFFFFFFFFu && pre_idx != post_idx && pre_idx < params.neuron_count) {
            let pre = neurons[pre_idx];
            let weight = compute_geometric_weight(pre, post);
            let contribution = weight * pre.voltage;
            geometric_input += contribution;
            geometric_sample_count += 1.0;
            
            if (abs(contribution) > abs(max_geometric_contrib)) {
                max_geometric_contrib = contribution;
                max_geometric_source = pre_idx;
            }
        }
    }
    
    if (geometric_sample_count > 0.0) {
        geometric_input /= geometric_sample_count;
        geometric_input *= f32(params.geometric_sample_count) * 10.0; 
    }
    
    let total_input = explicit_input + geometric_input * 0.5;
    
    // Dynamics
    post.voltage *= 0.95;
    post.voltage += total_input * params.dt;
    post.voltage = clamp(post.voltage, -2.0, 2.0);
    
    if (post.voltage > 1.0 && (params.time - post.spike_time) > post.refractory_period) {
        post.voltage = 0.0;
        post.spike_time = params.time;
    }
    
    // Learning
    if (params.train_mode == 1u) {
        let predicted = geometric_input * 0.5;
        let actual = post.voltage;
        let surprise = abs(actual - predicted);
        
        post.surprise_accumulator += surprise * params.dt;
        post.surprise_accumulator *= 0.99;
        
        if (surprise > params.terror_threshold && max_geometric_source != 0xFFFFFFFFu) {
            let existing_idx = find_explicit_synapse(&post, max_geometric_source);
            if (existing_idx >= 0) {
                let lr = kernel.explicit_learning_rate * (1.0 + surprise);
                post.explicit_weights[existing_idx] += lr * max_geometric_contrib * post.voltage;
                post.explicit_ages[existing_idx] = 0.0;
            } else if (abs(max_geometric_contrib) > kernel.promotion_threshold) {
                promote_to_explicit(&post, max_geometric_source, max_geometric_contrib * 2.0);
            }
        }
        
        for (var i = 0; i < 32; i++) {
            let pre_idx = post.explicit_targets[i];
            if (pre_idx != 0xFFFFFFFFu && pre_idx < params.neuron_count) {
                let pre = neurons[pre_idx];
                let correlation = pre.voltage * post.voltage;
                let lr = kernel.explicit_learning_rate * post.learning_rate;
                post.explicit_weights[i] += lr * correlation;
                
                if (abs(pre.voltage) > 0.1) {
                    post.explicit_ages[i] *= 0.95;
                }
            }
        }
        
        if (rand(post_idx + u32(params.time * 100.0)) < 0.01) {
            prune_explicit_synapses(&post);
        }
    }
    
    neurons[post_idx] = post;
}

@compute @workgroup_size(64)
fn cs_generate_lines(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    let neuron = neurons[idx];
    let start_offset = idx * params.explicit_synapse_slots * 2u;
    let origin = neuron.concept_pos * 3.0; 
    
    for (var i = 0u; i < params.explicit_synapse_slots; i++) {
        let target_idx = neuron.explicit_targets[i];
        let weight = neuron.explicit_weights[i];
        let buffer_idx = start_offset + i * 2u;
        
        if (target_idx != 0xFFFFFFFFu && target_idx < params.neuron_count && abs(weight) > 0.1) {
            let target_neuron = neurons[target_idx];
            let dest = target_neuron.concept_pos * 3.0;
            
            var col = vec3<f32>(0.0, 0.5, 1.0);
            if (weight < 0.0) { col = vec3<f32>(1.0, 0.0, 0.0); }
            
            let activity = (neuron.voltage + target_neuron.voltage) * 0.5;
            col *= (0.2 + activity * 2.0);

            line_buffer[buffer_idx].pos = origin;
            line_buffer[buffer_idx].color = col;
            line_buffer[buffer_idx+1].pos = dest;
            line_buffer[buffer_idx+1].color = col;
        } else {
            line_buffer[buffer_idx].pos = vec3<f32>(0.0);
            line_buffer[buffer_idx].color = vec3<f32>(0.0);
            line_buffer[buffer_idx+1].pos = vec3<f32>(0.0);
            line_buffer[buffer_idx+1].color = vec3<f32>(0.0);
        }
    }
}

// --- RENDERING ---

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
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
    let world_pos = neuron.concept_pos * 3.0 + vertex_pos * 0.04;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    let activity = neuron.voltage;
    let surprise = neuron.surprise_accumulator;
    let base = abs(neuron.concept_pos) * 0.3;
    let active_color = vec3<f32>(0.0, 1.0, 1.0) * activity;
    let terror_color = vec3<f32>(1.0, 0.0, 0.0) * surprise;
    
    out.color = base + active_color + terror_color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}

@vertex
fn vs_lines(
    @location(0) pos: vec3<f32>, 
    @location(1) col: vec3<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = col;
    return out;
}

@fragment
fn fs_lines(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    // Very low alpha to create the "cloud" effect
    return vec4<f32>(color, 0.15);
}