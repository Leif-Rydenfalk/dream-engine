// --- STRUCTS ---

struct Neuron {
    // GEOMETRIC & IDENTITY (0-48)
    concept_pos: vec3<f32>,          
    layer: u32,                      // 0=Retinal, 1=Cortex, 2=Hippocampus
    cortical_pos: vec2<f32>,         
    retinal_coord: vec2<u32>,        // XY coordinates for texture lookup
    receptive_center: vec3<f32>,     
    receptive_scale: f32,            
    
    // EXPLICIT SYNAPSES (48-560)
    explicit_targets: array<u32, 32>, 
    explicit_weights: array<f32, 32>,   
    explicit_ages: array<f32, 32>,
    explicit_visual_weights: array<f32, 32>,      
    
    // PLASTICITY & STATE (560-608)
    learning_rate: f32,
    homeostatic_target: f32,
    plasticity_trace: f32,           // Acts as short-term memory
    surprise_accumulator: f32,        
    voltage: f32,
    spike_time: f32,
    refractory_period: f32,
    top_geometric_contrib: f32,      
    top_geometric_source: u32,       
    
    // PADDING (Total size needs to be 16-byte aligned)
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

// Sensory Input
@group(0) @binding(6) var input_tex: texture_2d<f32>;
@group(0) @binding(7) var input_sampler: sampler;

// Cortex Visualization Output
@group(0) @binding(8) var output_tex: texture_storage_2d<rgba32float, write>;

// --- UTILITY ---

fn hash_to_grid(pos: vec3<f32>) -> u32 {
    // Expand the simulation bounds to [-4, 4] to catch edge cases
    let grid_size = f32(params.grid_dim);
    
    // Mapping [-2, 2] world space to [0, 1] for grid
    // (Since we multiply concept_pos by 3.0 in render, physics is roughly -1 to 1)
    let scaled = (pos + vec3<f32>(2.0)) / 4.0; 
    
    let cell = vec3<u32>(clamp(scaled * grid_size, vec3<f32>(0.0), vec3<f32>(grid_size - 1.0)));
    return cell.x + cell.y * params.grid_dim + cell.z * params.grid_dim * params.grid_dim;
}

fn rand(seed: u32) -> f32 {
    return fract(sin(f32(seed) * 12.9898) * 43758.5453);
}

// Quantized time randomizer to reduce geometric sampling jitter
fn rand_vec3_stable(seed: u32, time: f32) -> vec3<f32> {
    let t = floor(time * 20.0); // Update sampling pattern 20 times a second
    let s = f32(seed) + t;
    return vec3<f32>(
        fract(sin(s * 12.9898) * 43758.5453),
        fract(sin(s * 78.233) * 43758.5453),
        fract(sin(s * 44.123) * 43758.5453)
    ) * 2.0 - 1.0;
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
        (*post).explicit_visual_weights[slot] = 0.0; // Start invisible
        (*post).explicit_ages[slot] = 0.0;
    }
}

fn prune_explicit_synapses(post: ptr<function, Neuron>) {
    for (var i = 0; i < 32; i++) {
        (*post).explicit_ages[i] += params.dt;
        if ((*post).explicit_ages[i] > kernel.pruning_threshold) {
            (*post).explicit_targets[i] = 0xFFFFFFFFu;
            (*post).explicit_weights[i] = 0.0;
            (*post).explicit_visual_weights[i] = 0.0;
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
    
    // --- ADJUSTED FOR 500k TOTAL ---
    
    // 1. RETINA: 320x320 = 102,400 neurons
    // Enough resolution to track objects, but leaves room for brain.
    let retina_dim = 320u;
    let retina_count = retina_dim * retina_dim; 
    
    // 2. CORTEX: ~250,000 neurons
    // The main processing chunk
    let cortex_count = 250000u; 
    
    if (idx < retina_count) {
        // LAYER 0: RETINA
        n.layer = 0u;
        n.retinal_coord = vec2<u32>(idx % retina_dim, idx / retina_dim);
        
        // UV mapping
        let u = f32(n.retinal_coord.x) / f32(retina_dim);
        let v = f32(n.retinal_coord.y) / f32(retina_dim);
        
        n.cortical_pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.concept_pos = vec3<f32>(n.cortical_pos, -0.8); 
        n.learning_rate = 0.0; 
    } 
    else if (idx < retina_count + cortex_count) {
        // LAYER 1: CORTEX
        n.layer = 1u;
        n.retinal_coord = vec2<u32>(0u, 0u);
        
        // Spread them out well in the center
        n.cortical_pos = vec2<f32>(rand(seed), rand(seed+1u)) * 2.0 - 1.0;
        n.concept_pos = vec3<f32>(n.cortical_pos, 0.0 + (rand(seed+2u)*0.4 - 0.2));
        n.learning_rate = 0.02; 
    } 
    else {
        // LAYER 2: HIPPOCAMPUS (Remainder)
        // ~147,600 neurons left
        n.layer = 2u;
        n.retinal_coord = vec2<u32>(0u, 0u);
        
        n.cortical_pos = vec2<f32>(rand(seed), rand(seed+1u)) * 2.0 - 1.0;
        n.concept_pos = vec3<f32>(n.cortical_pos, 0.8 + (rand(seed+2u)*0.4 - 0.2));
        n.learning_rate = 0.005; 
    }

    n.receptive_center = vec3<f32>(rand(seed + 20u), rand(seed + 21u), rand(seed + 22u)) * 2.0 - 1.0;
    n.receptive_scale = 0.5 + rand(seed + 30u) * 0.5;
    
    for (var i = 0; i < 32; i++) {
        n.explicit_targets[i] = 0xFFFFFFFFu;
        n.explicit_weights[i] = 0.0;
        n.explicit_visual_weights[i] = 0.0;
        n.explicit_ages[i] = 0.0;
    }
    
    n.homeostatic_target = 0.2;
    n.voltage = 0.0;
    n.refractory_period = 0.05;
    n.plasticity_trace = 0.0;
    n.surprise_accumulator = 0.0;
    
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

    // ==========================================
    // PHASE 1: SENSORY INPUT & DREAMING
    // ==========================================
    
    if (post.layer == 0u) {
        // SENSING MODE (Awake)
        if (params.train_mode == 1u) {
            // Use exact integer coordinates for crisp input
            let val = textureLoad(input_tex, post.retinal_coord, 0).r;
            
            post.voltage = (val - 0.5) * 3.0; // Increased contrast
            post.voltage = clamp(post.voltage, -1.0, 1.0);
            post.plasticity_trace = post.voltage;
            
            neurons[post_idx] = post;
            return; 
        }
        // DREAM MODE (Asleep)
        else {
            // Top-Down Reconstruction:
            // The Retina looks "UP" at the Cortex (Layer 1) to see what it is imagining.
            // This inverts the geometric sampling.
            
            var dream_input = 0.0;
            let dream_samples = 16u; // Lower sample count for retina to save perf
            
            for (var i = 0u; i < dream_samples; i++) {
                // Look at Z = 0.0 (Cortex Layer)
                let offset = rand_vec3_stable(post_idx * 50u + i, params.time) * 0.2;
                let probe_pos = vec3<f32>(post.cortical_pos, 0.0) + offset;
                
                let grid_idx = hash_to_grid(probe_pos);
                let pre_idx = atomicLoad(&spatial_grid[grid_idx]);
                
                if (pre_idx != 0xFFFFFFFFu && pre_idx < params.neuron_count) {
                    let pre = neurons[pre_idx];
                    // Only listen to Cortex (Layer 1)
                    if (pre.layer == 1u) {
                        // Distance weighting
                        let dist = distance(vec3<f32>(post.cortical_pos, 0.0), pre.concept_pos);
                        let weight = exp(-dist * dist * 15.0);
                        
                        // Add the cortex's activity to the retinal pixel
                        dream_input += pre.voltage * weight * 3.0;
                    }
                }
            }
            
            // Smooth transition into dream state (Temporal anti-aliasing)
            post.voltage += (tanh(dream_input) - post.voltage) * 0.1;
            post.plasticity_trace = post.voltage;
            
            neurons[post_idx] = post;
            return;
        }
    }

    // ==========================================
    // PHASE 2: SYNAPTIC INTEGRATION
    // ==========================================

    var input_current = 0.0;
    var geometric_max = 0.0;
    var geometric_source = 0xFFFFFFFFu;

    // 2a. Explicit Synapses (Long-term memory)
    for (var i = 0; i < 32; i++) {
        let pre_idx = post.explicit_targets[i];
        if (pre_idx == 0xFFFFFFFFu) { continue; }
        
        let pre = neurons[pre_idx];
        let w = post.explicit_weights[i];
        
        // Standard integrate
        input_current += w * pre.voltage;
        
        // Decay ages
        post.explicit_ages[i] += params.dt;
        
        // Visual smoothing
        let target_vis = select(0.0, abs(w), abs(w) > 0.1);
        post.explicit_visual_weights[i] += (target_vis - post.explicit_visual_weights[i]) * 0.1;
    }

    // 2b. Geometric Sampling (The "Dendrites")
    // We scan the spatial grid. We filter based on hierarchy.
    let local_samples = params.geometric_sample_count;
    
    for (var i = 0u; i < local_samples; i++) {
        // Jitter probe position based on layer logic
        // Cortex (L1) looks "down" at Retina (Z = -0.8)
        // Hippo (L2) looks "down" at Cortex (Z = 0.0)
        
        var look_target_z = 0.0;
        if (post.layer == 1u) { look_target_z = -0.8; } // Look at retina
        if (post.layer == 2u) { look_target_z = 0.0; }  // Look at cortex
        
        // Probe around the projected cortical position
        let offset = rand_vec3_stable(post_idx * 200u + i, params.time) * 0.15;
        let probe_pos = vec3<f32>(post.cortical_pos, look_target_z) + offset;
        
        let grid_idx = hash_to_grid(probe_pos);
        let pre_idx = atomicLoad(&spatial_grid[grid_idx]);
        
        if (pre_idx != 0xFFFFFFFFu && pre_idx != post_idx && pre_idx < params.neuron_count) {
            let pre = neurons[pre_idx];
            
            // HIERARCHY RULE: Only accept input from the layer below or same layer
            let layer_diff = i32(post.layer) - i32(pre.layer);
            
            var valid_connection = false;
            var weight_scale = 1.0;

            // Feedforward: Layer N reads Layer N-1
            if (layer_diff == 1) {
                valid_connection = true;
                weight_scale = 1.5; // Strong bottom-up drive
            }
            // Lateral: Layer N reads Layer N (Lateral inhibition/excitation)
            else if (layer_diff == 0) {
                valid_connection = true;
                weight_scale = 0.5; // Weaker lateral connection
            }

            if (valid_connection) {
                let dist = distance(pre.concept_pos, probe_pos); // Use probe pos for "receptive field"
                let geo_weight = exp(-dist * dist * 10.0) * weight_scale; 
                
                let signal = geo_weight * pre.voltage;
                input_current += signal;
                
                // Track strongest input for learning
                if (abs(signal) > abs(geometric_max)) {
                    geometric_max = signal;
                    geometric_source = pre_idx;
                }
            }
        }
    }
    
    // Normalize geometric input
    input_current *= 2.0;

    // ==========================================
    // PHASE 3: DYNAMICS & SPIKING
    // ==========================================

    // Leak
    post.voltage *= 0.92; 
    
    // Integrate
    post.voltage += input_current * params.dt;
    
    // Activation Function (Tanh-like soft clamp)
    post.voltage = tanh(post.voltage);

    // Update Trace (Low-pass filter of activity)
    // This represents Ca2+ concentration or recent firing rate
    post.plasticity_trace += (abs(post.voltage) - post.plasticity_trace) * 0.1;

    // ==========================================
    // PHASE 4: LEARNING (The "Dream" Logic)
    // ==========================================
    
    if (params.train_mode == 1u) {
        // 4a. Structural Plasticity (Forming new explicit connections)
        // If we are highly active (voltage high) AND we have a strong geometric input
        // that isn't explicit yet, make it explicit.
        if (abs(post.voltage) > 0.5 && abs(geometric_max) > kernel.promotion_threshold) {
            // Check if we already have it
            let existing_idx = find_explicit_synapse(&post, geometric_source);
            
            if (existing_idx == -1) {
                 promote_to_explicit(&post, geometric_source, geometric_max * 0.5);
                 post.surprise_accumulator += 1.0; // Visual flair for "Learning Event"
            }
        }
        
        // 4b. Hebbian Learning (Fire together, wire together)
        for (var i = 0; i < 32; i++) {
            let pre_idx = post.explicit_targets[i];
            if (pre_idx == 0xFFFFFFFFu) { continue; }
            
            let pre = neurons[pre_idx];
            
            // Pre-Post correlation
            // If Post is firing (Voltage) and Pre fired recently (Trace), strengthen.
            let hebbian = post.voltage * pre.plasticity_trace; 
            
            // Weight update
            post.explicit_weights[i] += hebbian * post.learning_rate;
            
            // Normalization/Decay (prevent explosion)
            post.explicit_weights[i] *= 0.999;
            
            // Reset age if useful
            if (abs(hebbian) > 0.01) {
                post.explicit_ages[i] = 0.0;
            }
        }
        
        // 4c. Pruning
        if (id.x % 100u == 0u) { // Don't prune every frame, expensive
            prune_explicit_synapses(&post);
        }
    }
    
    post.surprise_accumulator *= 0.95; // Decay visual flair

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
        let vis_weight = neuron.explicit_visual_weights[i]; 
        let buffer_idx = start_offset + i * 2u;
        
        if (target_idx != 0xFFFFFFFFu && target_idx < params.neuron_count && vis_weight > 0.05) {
            let target_neuron = neurons[target_idx];
            let dest = target_neuron.concept_pos * 3.0;
            
            let weight = neuron.explicit_weights[i];
            var col = vec3<f32>(0.0, 0.8, 1.0); 
            if (weight < 0.0) { col = vec3<f32>(1.0, 0.0, 0.2); } 
            
            let activity = (abs(neuron.voltage) + abs(target_neuron.voltage)) * 0.5;
            col *= (vis_weight + activity * 2.0);

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

@compute @workgroup_size(8, 8)
fn cs_render_cortex(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    let uv = vec2<f32>(f32(id.x), f32(id.y)) / vec2<f32>(dim);
    let cortical_pos = uv * 2.0 - 1.0; // [-1, 1]
    
    let cortical_pos_3d = vec3<f32>(cortical_pos, 0.0);
    let grid_idx = hash_to_grid(cortical_pos_3d);
    let n_idx = atomicLoad(&spatial_grid[grid_idx]);
    
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    if (n_idx < params.neuron_count) {
        let n = neurons[n_idx];
        if (distance(n.cortical_pos, cortical_pos) < 0.05) {
            let v = n.voltage;
            let s = n.surprise_accumulator;
            
            // Heatmap: Blue = Resting, Cyan = Active, Red = Surprise
            let base = vec3<f32>(0.1, 0.1, 0.2);
            // RENAME FIX: "active" -> "activity_color"
            let activity_color = vec3<f32>(0.0, 1.0, 1.0) * max(0.0, v); 
            let surprise = vec3<f32>(1.0, 0.0, 0.0) * s * 5.0;
            
            color = vec4<f32>(base + activity_color + surprise, 1.0);
        }
    }
    
    textureStore(output_tex, vec2<i32>(id.xy), color);
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
    
    // Reduce size: 0.01 instead of 0.04
    let world_pos = neuron.concept_pos * 3.0 + vertex_pos * 0.01; 
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    let activity = neuron.voltage;
    let trace = neuron.plasticity_trace; // Visualize the trace
    
    // Layer Colors
    var base_color = vec3<f32>(0.1);
    if (neuron.layer == 0u) { base_color = vec3<f32>(0.1, 0.1, 0.1); } // Retina (Dark)
    if (neuron.layer == 1u) { base_color = vec3<f32>(0.1, 0.0, 0.1); } // Cortex (Purple base)
    if (neuron.layer == 2u) { base_color = vec3<f32>(0.0, 0.1, 0.1); } // Hippo (Teal base)

    let active_color = vec3<f32>(0.0, 1.0, 1.0) * max(0.0, activity);
    // Show recent memory (trace) as Green
    let trace_color = vec3<f32>(0.0, 1.0, 0.0) * trace * 0.5; 
    
    out.color = base_color + active_color + trace_color;
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
    return vec4<f32>(color, 0.2);
}