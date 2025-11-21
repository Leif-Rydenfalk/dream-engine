// --- STRUCTS ---

struct Neuron {
    // GEOMETRIC & IDENTITY (0-48)
    concept_pos: vec3<f32>,          
    layer: u32,                      // 0=Retinal, 1=Cortex, 2=Hippocampus
    cortical_pos: vec2<f32>,         
    retinal_coord: vec2<u32>,        
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
    plasticity_trace: f32,           // Holds PREDICTION (Retina) or CONFIDENCE (Cortex)
    surprise_accumulator: f32,       // Holds ABSOLUTE ERROR
    voltage: f32,                    // Holds SIGNED ERROR (Retina) or ACTIVITY (Cortex)
    spike_time: f32,
    refractory_period: f32,
    top_geometric_contrib: f32,      
    top_geometric_source: u32,       
    
    // PADDING
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
    train_mode: u32, // 1 = Learn, 0 = Inference
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

// Cortex Error Output (Viz 1)
@group(0) @binding(8) var output_tex: texture_storage_2d<rgba32float, write>;

// Cortex Prediction/Dream Output (Viz 2)
@group(0) @binding(9) var prediction_tex: texture_storage_2d<rgba32float, write>;

// --- UTILITY ---

fn hash_to_grid(pos: vec3<f32>) -> u32 {
    let grid_size = f32(params.grid_dim);
    // Mapping [-2, 2] world space to [0, 1] for grid
    let scaled = (pos + vec3<f32>(2.0)) / 4.0; 
    let cell = vec3<u32>(clamp(scaled * grid_size, vec3<f32>(0.0), vec3<f32>(grid_size - 1.0)));
    return cell.x + cell.y * params.grid_dim + cell.z * params.grid_dim * params.grid_dim;
}

fn rand(seed: u32) -> f32 {
    return fract(sin(f32(seed) * 12.9898) * 43758.5453);
}

fn rand_vec3_stable(seed: u32, time: f32) -> vec3<f32> {
    let t = floor(time * 20.0); 
    let s = f32(seed) + t;
    return vec3<f32>(
        fract(sin(s * 12.9898) * 43758.5453),
        fract(sin(s * 78.233) * 43758.5453),
        fract(sin(s * 44.123) * 43758.5453)
    ) * 2.0 - 1.0;
}

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
        (*post).explicit_visual_weights[slot] = 0.0; 
        (*post).explicit_ages[slot] = 0.0;
    }
}

// --- COMPUTE SHADERS ---

@compute @workgroup_size(64)
fn cs_init_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.neuron_count) { return; }
    
    var n: Neuron;
    let seed = idx * 1000u;
    
    let retina_dim = 320u;
    let retina_count = retina_dim * retina_dim; 
    let cortex_count = 250000u; 
    
    if (idx < retina_count) {
        // LAYER 0: RETINA (Error Detector)
        n.layer = 0u;
        n.retinal_coord = vec2<u32>(idx % retina_dim, idx / retina_dim);
        let u = f32(n.retinal_coord.x) / f32(retina_dim);
        let v = f32(n.retinal_coord.y) / f32(retina_dim);
        n.cortical_pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
        n.concept_pos = vec3<f32>(n.cortical_pos, -0.8); 
        n.learning_rate = 0.0; 
    } 
    else if (idx < retina_count + cortex_count) {
        // LAYER 1: CORTEX (Predictor)
        n.layer = 1u;
        n.retinal_coord = vec2<u32>(0u, 0u);
        n.cortical_pos = vec2<f32>(rand(seed), rand(seed+1u)) * 2.0 - 1.0;
        n.concept_pos = vec3<f32>(n.cortical_pos, 0.0 + (rand(seed+2u)*0.4 - 0.2));
        n.learning_rate = 0.02; 
    } 
    else {
        // LAYER 2: HIPPOCAMPUS (Context)
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
    
    // Cortex needs to be visible at Z=0 for the Retina to find it easily
    if (n.layer == 1u) {
        let cortical_hash = hash_to_grid(vec3<f32>(n.cortical_pos, 0.0));
        atomicStore(&spatial_grid[cortical_hash], idx);
    }
}

@compute @workgroup_size(64)
fn cs_update_neurons(@builtin(global_invocation_id) id: vec3<u32>) {
    let post_idx = id.x;
    if (post_idx >= params.neuron_count) { return; }
    
    var post = neurons[post_idx];

    // =====================================================
    // LOGIC 1: RETINA (THE ERROR DETECTOR)
    // =====================================================
    if (post.layer == 0u) {
        // 1. Get Reality (Bottom-Up)
        let reality = textureLoad(input_tex, post.retinal_coord, 0).r;
        let reality_val = (reality - 0.5) * 3.0; // Normalize centered on 0

        // 2. Get Prediction (Top-Down from Cortex)
        var prediction_sum = 0.0;
        var prediction_count = 0.0;
        
        let dream_samples = 12u; 
        
        for (var i = 0u; i < dream_samples; i++) {
            // Cortex is at Z=0.0. Sample around there.
            let offset = rand_vec3_stable(post_idx * 50u + i, params.time) * 0.15;
            let probe_pos = vec3<f32>(post.cortical_pos, 0.0) + offset;
            
            let grid_idx = hash_to_grid(probe_pos);
            let pre_idx = atomicLoad(&spatial_grid[grid_idx]);
            
            if (pre_idx != 0xFFFFFFFFu && pre_idx < params.neuron_count) {
                let pre = neurons[pre_idx];
                if (pre.layer == 1u) {
                    let dist = distance(vec3<f32>(post.cortical_pos, 0.0), pre.concept_pos);
                    let weight = exp(-dist * dist * 20.0);
                    
                    // Cortex Activity (Voltage) IS the Prediction
                    prediction_sum += pre.voltage * weight;
                    prediction_count += weight;
                }
            }
        }
        
        var prediction = 0.0;
        if (prediction_count > 0.001) {
            prediction = prediction_sum / prediction_count;
        }
        
        // Smooth prediction into trace for visualization
        post.plasticity_trace = mix(post.plasticity_trace, prediction, 0.2);
        
        // 3. Compute Error = Reality - Prediction
        let error = reality_val - post.plasticity_trace;
        
        // 4. Set State
        post.voltage = error; // Send error up
        post.surprise_accumulator = mix(post.surprise_accumulator, abs(error), 0.1);
        
        neurons[post_idx] = post;
        return;
    }

    // =====================================================
    // LOGIC 2: CORTEX (THE PREDICTION ENGINE)
    // =====================================================
    
    var input_error = 0.0;
    var max_error_source = 0xFFFFFFFFu;
    var max_error_val = 0.0;

    // 2a. Lateral Connections (Explicit Synapses)
    for (var i = 0; i < 32; i++) {
        let pre_idx = post.explicit_targets[i];
        if (pre_idx == 0xFFFFFFFFu) { continue; }
        
        let pre = neurons[pre_idx];
        let w = post.explicit_weights[i];
        
        // Standard recurrent integration
        input_error += w * pre.voltage; 
        post.explicit_ages[i] += params.dt;
        
        let target_vis = select(0.0, abs(w), abs(w) > 0.1);
        post.explicit_visual_weights[i] += (target_vis - post.explicit_visual_weights[i]) * 0.1;
    }

    // 2b. Sample Bottom-Up Error from Retina
    let local_samples = params.geometric_sample_count;
    
    for (var i = 0u; i < local_samples; i++) {
        // Retina is at Z = -0.8
        let offset = rand_vec3_stable(post_idx * 200u + i, params.time) * 0.15;
        let probe_pos = vec3<f32>(post.cortical_pos, -0.8) + offset; 
        
        let grid_idx = hash_to_grid(probe_pos);
        let pre_idx = atomicLoad(&spatial_grid[grid_idx]);
        
        if (pre_idx != 0xFFFFFFFFu && pre_idx < params.neuron_count) {
            let pre = neurons[pre_idx];
            
            // Only listen to Retina
            if (pre.layer == 0u) {
                let dist = distance(probe_pos, pre.concept_pos);
                let weight = exp(-dist * dist * 10.0);
                
                // Pre.voltage is ERROR
                let incoming_error = pre.voltage * weight;
                
                input_error += incoming_error * 2.0; 
                
                if (abs(incoming_error) > abs(max_error_val)) {
                    max_error_val = incoming_error;
                    max_error_source = pre_idx;
                }
            }
        }
    }

    // 2c. Dynamics
    post.voltage *= 0.92; // Leak
    post.voltage += input_error * params.dt;
    post.voltage = clamp(post.voltage, -1.0, 1.0);
    
    // Trace stores confidence/stable state
    post.plasticity_trace += (post.voltage - post.plasticity_trace) * 0.1;
    
    // 2d. LEARNING (Delta Rule / Free Energy Minimization)
    if (params.train_mode == 1u) {
        
        let learning_rate = post.learning_rate * kernel.explicit_learning_rate;
        
        for (var i = 0; i < 32; i++) {
            let pre_idx = post.explicit_targets[i];
            if (pre_idx == 0xFFFFFFFFu) { continue; }
            let pre = neurons[pre_idx];
            
            // Delta Rule: Change weight to minimize future error.
            // Weight += Error * Pre_Input
            // Here, 'input_error' acts as the global error gradient for this neuron
            let delta = input_error * pre.plasticity_trace; 
            
            post.explicit_weights[i] += delta * learning_rate;
            post.explicit_weights[i] *= 0.9995; // Decay
            
            // Structural Plasticity - promote connections that reduce error?
            // For now, we just prune very old connections
            if (post.explicit_ages[i] > kernel.pruning_threshold) {
                // In standard Hebbian, we prune weak weights.
                // In predictive coding, a 0 weight means no prediction dependency.
                 if (abs(post.explicit_weights[i]) < 0.05) {
                     post.explicit_targets[i] = 0xFFFFFFFFu;
                     post.explicit_weights[i] = 0.0;
                 }
            }
        }
        
        // Structural Plasticity: Add connection if we have high activity but few connections
        if (abs(post.voltage) > 0.5 && rand(id.x + u32(params.time*10.0)) > 0.98) {
             // In a full model, we'd search for correlated neurons.
             // Here, rely on init density.
        }
    }
    
    post.surprise_accumulator = mix(post.surprise_accumulator, abs(input_error), 0.05);

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
            var col = vec3<f32>(0.0, 0.8, 1.0); // Excitatory (Cyan)
            if (weight < 0.0) { col = vec3<f32>(1.0, 0.0, 0.2); } // Inhibitory (Red)
            
            // Visualize learning: Pulse brightness on high error
            let activity = neuron.surprise_accumulator;
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

// --- VISUALIZERS ---

@compute @workgroup_size(8, 8)
fn cs_render_dream(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    // Map screen UV to Retinal ID
    let uv = vec2<f32>(id.xy) / vec2<f32>(dim);
    let retinal_coord = vec2<u32>(u32(uv.x * 320.0), u32(uv.y * 320.0));
    let idx = retinal_coord.x + retinal_coord.y * 320u;
    
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    if (idx < params.neuron_count) {
        let n = neurons[idx];
        // plasticity_trace holds the Top-Down Prediction
        let pred = n.plasticity_trace * 0.5 + 0.5; 
        color = vec4<f32>(pred, pred, pred, 1.0);
    }
    
    textureStore(prediction_tex, vec2<i32>(id.xy), color);
}

@compute @workgroup_size(8, 8)
fn cs_render_cortex(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<u32>(512, 512);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    
    let uv = vec2<f32>(f32(id.x), f32(id.y)) / vec2<f32>(dim);
    let cortical_pos = uv * 2.0 - 1.0;
    
    let cortical_pos_3d = vec3<f32>(cortical_pos, 0.0);
    let grid_idx = hash_to_grid(cortical_pos_3d);
    let n_idx = atomicLoad(&spatial_grid[grid_idx]);
    
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    if (n_idx < params.neuron_count) {
        let n = neurons[n_idx];
        if (distance(n.cortical_pos, cortical_pos) < 0.03) {
            let activity = n.voltage; // Error
            let confidence = n.plasticity_trace; // State
            
            // Red = High Error (Surprise)
            let error_col = vec3<f32>(1.0, 0.2, 0.0) * max(0.0, activity);
            // Purple = Negative Error 
            let neg_error_col = vec3<f32>(0.8, 0.0, 1.0) * max(0.0, -activity);
            
            // Cyan = Stable State
            let stable_col = vec3<f32>(0.0, 0.5, 1.0) * abs(confidence);
            
            color = vec4<f32>(error_col + neg_error_col + stable_col, 1.0);
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
    let world_pos = neuron.concept_pos * 3.0 + vertex_pos * 0.01; 
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    let error = neuron.voltage; // Error signal
    let pred = neuron.plasticity_trace; // Prediction state
    
    var base_color = vec3<f32>(0.05);
    
    // Retina: Visualize Error
    if (neuron.layer == 0u) { 
        // White = Zero Error (Good Match)
        // Red = Positive Error, Blue = Negative Error
        let err = error; 
        base_color = vec3<f32>(0.1) + vec3<f32>(err, 0.0, -err);
    } 
    // Cortex: Visualize Prediction + Surprise
    else if (neuron.layer == 1u) { 
        // Cyan = Prediction Strength
        // Red Flash = Surprise (Error Recieved)
        let surprise = neuron.surprise_accumulator * 2.0;
        base_color = vec3<f32>(surprise, abs(pred)*0.5, abs(pred));
    }

    out.color = base_color;
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