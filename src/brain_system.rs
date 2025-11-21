use crate::{vertex, CameraFeed, ShaderHotReload};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// Update constants at the top
// const NEURON_COUNT: u32 = 1_048_576; // 1 Million (1024 x 1024)
const NEURON_COUNT: u32 = 448_576;
const GRID_DIM: u32 = 128; // Finer grid for more neurons
const EXPLICIT_SLOTS: usize = 32; // Keep at 32 for now (1M * 32 = 32 Million synapses!)

// --- GPU STRUCTS ---
// MUST match the WGSL struct padding exactly (16-byte alignment)

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Neuron {
    // GEOMETRIC (0-48 bytes)
    pub concept_pos: [f32; 3],
    pub layer: u32, // Modified: pad1 replaced by layer
    pub cortical_pos: [f32; 2],
    pub retinal_coord: [u32; 2], // Modified: pad2 replaced by coordinates
    pub receptive_center: [f32; 3],
    pub receptive_scale: f32,

    // EXPLICIT (48-560 bytes)
    pub explicit_targets: [u32; EXPLICIT_SLOTS],
    pub explicit_weights: [f32; EXPLICIT_SLOTS],
    pub explicit_ages: [f32; EXPLICIT_SLOTS],
    pub explicit_visual_weights: [f32; EXPLICIT_SLOTS],

    // PLASTICITY & STATE (560-608 bytes)
    pub learning_rate: f32,
    pub homeostatic_target: f32,
    pub plasticity_trace: f32,
    pub surprise_accumulator: f32,
    pub voltage: f32,
    pub spike_time: f32,
    pub refractory_period: f32,
    pub top_geometric_contrib: f32,
    pub top_geometric_source: u32,

    // PADDING (To reach 608 or 16-byte align)
    pub _padding_final: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SynapticKernel {
    pub local_amplitude: f32,
    pub local_decay: f32,
    pub conceptual_amplitude: f32,
    pub conceptual_decay: f32,
    pub inhibit_radius: f32,
    pub inhibit_strength: f32,
    pub explicit_learning_rate: f32,
    pub pruning_threshold: f32,
    pub promotion_threshold: f32,
    pub temporal_decay: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub neuron_count: u32,
    pub time: f32,
    pub dt: f32,
    pub geometric_sample_count: u32,

    pub explicit_synapse_slots: u32,
    pub train_mode: u32,
    pub terror_threshold: f32,
    pub grid_dim: u32,

    pub _pad: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LineVertex {
    pos: [f32; 3],
    _pad: f32,
    color: [f32; 3],
    _pad2: f32,
}

pub struct BrainSystem {
    queue: Arc<wgpu::Queue>,

    // Buffers
    neuron_buffer: wgpu::Buffer,
    param_buffer: wgpu::Buffer,
    kernel_buffer: wgpu::Buffer,
    spatial_grid_buffer: wgpu::Buffer,
    spike_history_buffer: wgpu::Buffer,
    synapse_line_buffer: wgpu::Buffer,
    vertex_buffer_cube: wgpu::Buffer,
    index_buffer_cube: wgpu::Buffer,

    // Textures
    pub input_texture: Arc<wgpu::Texture>,
    pub input_texture_view: Arc<wgpu::TextureView>,
    pub output_texture: Arc<wgpu::Texture>,
    pub output_texture_view: Arc<wgpu::TextureView>,

    pub camera: CameraFeed,

    // Bind Groups
    compute_bind_group: wgpu::BindGroup,
    render_points_bind_group: wgpu::BindGroup,
    empty_bind_group: wgpu::BindGroup,

    // Pipelines
    init_pipeline: wgpu::ComputePipeline,
    clear_grid_pipeline: wgpu::ComputePipeline,
    populate_grid_pipeline: wgpu::ComputePipeline,
    update_neurons_pipeline: wgpu::ComputePipeline,
    generate_lines_pipeline: wgpu::ComputePipeline,
    render_cortex_pipeline: wgpu::ComputePipeline,

    render_pipeline_points: wgpu::RenderPipeline,
    render_pipeline_lines: wgpu::RenderPipeline,

    // State
    pub params: SimParams,
    pub kernel: SynapticKernel,
    start_time: std::time::Instant,
    initialized: bool,
}

impl BrainSystem {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_hot_reload: &ShaderHotReload,
        camera_url: String,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // 1. Define Initial State
        let params = SimParams {
            neuron_count: NEURON_COUNT,
            time: 0.0,
            dt: 0.016,
            geometric_sample_count: 32, // Reduce slightly to 32 to maintain 60fps with 1M neurons
            explicit_synapse_slots: EXPLICIT_SLOTS as u32,
            train_mode: 1,
            terror_threshold: 0.5,
            grid_dim: GRID_DIM,
            _pad: [0.0; 4],
        };

        let kernel = SynapticKernel {
            local_amplitude: 1.5,
            local_decay: 0.1,
            conceptual_amplitude: 0.8,
            conceptual_decay: 2.0,
            inhibit_radius: 0.1,
            inhibit_strength: 0.5,
            explicit_learning_rate: 0.05,
            pruning_threshold: 5.0,
            promotion_threshold: 0.3,
            temporal_decay: 0.05,
            _pad: [0.0; 2],
        };

        // 2. Create Buffers
        let neuron_buffer_size = (std::mem::size_of::<Neuron>() * NEURON_COUNT as usize) as u64;
        let neuron_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neuron Buffer"),
            size: neuron_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let kernel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Synaptic Kernel"),
            contents: bytemuck::cast_slice(&[kernel]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_len = GRID_DIM * GRID_DIM * GRID_DIM;
        let spatial_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Grid"),
            size: (grid_len as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Unused in new logic but kept for structure compatibility if needed
        let spike_history_len = NEURON_COUNT * 100;
        let spike_history_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spike History"),
            size: (spike_history_len as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let max_lines = NEURON_COUNT as usize * EXPLICIT_SLOTS;
        let line_buffer_size = (max_lines * 2 * std::mem::size_of::<LineVertex>()) as u64;
        let synapse_line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Synapse Lines"),
            size: line_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let vertex_buffer_cube = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertices"),
            contents: bytemuck::cast_slice(vertex::VERTICES_CUBE),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Index Buffer for Cube
        let mut cube_indices: Vec<u16> = Vec::new();
        for i in 0..6 {
            let base = (i * 4) as u16;
            cube_indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        let index_buffer_cube = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Indices"),
            contents: bytemuck::cast_slice(&cube_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // 3. Textures & Samplers
        let tex_size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };

        // Input Texture (From Camera)
        let input_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input Tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let input_texture_view = Arc::new(input_texture.create_view(&Default::default()));

        let input_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Input Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Output Texture (For 2D Visualization of Cortex)
        let output_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        }));
        let output_texture_view = Arc::new(output_texture.create_view(&Default::default()));

        // 4. Bind Groups
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Brain Compute Layout"),
            entries: &[
                // 0: Neurons
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: Params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Kernel
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Grid
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: History
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: Lines
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: Input Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 7: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 8: Output Texture (Storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brain Compute BG"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: neuron_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spatial_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spike_history_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: synapse_line_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&input_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&input_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
            ],
        });

        let render_points_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Points Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let render_points_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Points BG"),
            layout: &render_points_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: neuron_buffer.as_entire_binding(),
            }],
        });

        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Empty Layout"),
            entries: &[],
        });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Empty BG"),
            layout: &empty_layout,
            entries: &[],
        });

        // 5. Pipelines
        let shader_module = shader_hot_reload.get_shader("brain.wgsl");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[&compute_layout],
            push_constant_ranges: &[],
        });

        // Helper to create pipelines quickly
        let create_compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let init_pipeline = create_compute("cs_init_neurons");
        let clear_grid_pipeline = create_compute("cs_clear_grid");
        let populate_grid_pipeline = create_compute("cs_populate_grid");
        let update_neurons_pipeline = create_compute("cs_update_neurons");
        let generate_lines_pipeline = create_compute("cs_generate_lines");
        let render_cortex_pipeline = create_compute("cs_render_cortex");

        // Render Pipelines
        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };

        let depth_stencil_state = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Points Pipeline (Neuron Cubes)
        let render_pipeline_points_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Points Pipeline Layout"),
                bind_group_layouts: &[&render_points_layout, camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_points =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Point Cloud Pipeline"),
                layout: Some(&render_pipeline_points_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex::create_vertex_buffer_layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(additive_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: depth_stencil_state.clone(),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        // Lines Pipeline (Synapses)
        let line_render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Line Render Layout"),
            bind_group_layouts: &[&empty_layout, camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline_lines =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Line Pipeline"),
                layout: Some(&line_render_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_lines"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 16,
                                shader_location: 1,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_lines"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(additive_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: depth_stencil_state,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        Self {
            queue,
            neuron_buffer,
            param_buffer,
            kernel_buffer,
            spatial_grid_buffer,
            spike_history_buffer,
            synapse_line_buffer,
            vertex_buffer_cube,
            index_buffer_cube,
            input_texture,
            input_texture_view,
            output_texture,
            output_texture_view,
            camera: CameraFeed::new(camera_url),
            compute_bind_group,
            render_points_bind_group,
            empty_bind_group,
            init_pipeline,
            clear_grid_pipeline,
            populate_grid_pipeline,
            update_neurons_pipeline,
            generate_lines_pipeline,
            render_cortex_pipeline,
            render_pipeline_points,
            render_pipeline_lines,
            params,
            kernel,
            start_time: std::time::Instant::now(),
            initialized: false,
        }
    }

    pub fn update(&mut self, input: &crate::input::Input, _window_size: (u32, u32)) {
        self.params.time = self.start_time.elapsed().as_secs_f32();

        // Toggle Training Mode (Spacebar)
        // Mode 1 = Awake (Learning from Camera)
        // Mode 0 = Dreaming (Generative / Top-Down)
        if input.is_key_pressed(winit::keyboard::KeyCode::Space) {
            self.params.train_mode = if self.params.train_mode == 1 { 0 } else { 1 };
            println!(
                "Mode switched: {}",
                if self.params.train_mode == 1 {
                    "AWAKE"
                } else {
                    "DREAMING"
                }
            );
        }

        // Upload Camera Frame
        if let Some(img) = self.camera.get_frame() {
            // Resize to match texture dimensions
            let img = image::imageops::resize(&img, 512, 512, image::imageops::FilterType::Nearest);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &img,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * 512),
                    rows_per_image: Some(512),
                },
                wgpu::Extent3d {
                    width: 512,
                    height: 512,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Update Uniforms
        self.queue
            .write_buffer(&self.param_buffer, 0, bytemuck::cast_slice(&[self.params]));
        self.queue
            .write_buffer(&self.kernel_buffer, 0, bytemuck::cast_slice(&[self.kernel]));
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bg: &wgpu::BindGroup,
        depth_view: &wgpu::TextureView,
        target: &wgpu::TextureView,
    ) {
        // --- COMPUTE PASS ---
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Brain Compute"),
                timestamp_writes: None,
            });

            if !self.initialized {
                cpass.set_pipeline(&self.init_pipeline);
                cpass.set_bind_group(0, &self.compute_bind_group, &[]);
                cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);
                self.initialized = true;
            }

            // 1. Clear Grid
            cpass.set_pipeline(&self.clear_grid_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            let grid_len = GRID_DIM * GRID_DIM * GRID_DIM;
            cpass.dispatch_workgroups((grid_len + 63) / 64, 1, 1);

            // 2. Populate Grid (Spatial Hashing)
            cpass.set_pipeline(&self.populate_grid_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);

            // 3. Update Neurons (Sensory -> Logic -> Plasticity)
            cpass.set_pipeline(&self.update_neurons_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);

            // 4. Generate Visualization Lines
            cpass.set_pipeline(&self.generate_lines_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);

            // 5. Render 2D Cortex View (Optional, for debugging texture)
            cpass.set_pipeline(&self.render_cortex_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(512 / 8, 512 / 8, 1);
        }

        // --- RENDER PASS ---
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Brain Draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Draw Lines (Synapses)
            rpass.set_pipeline(&self.render_pipeline_lines);
            rpass.set_bind_group(0, &self.empty_bind_group, &[]);
            rpass.set_bind_group(1, camera_bg, &[]);
            rpass.set_vertex_buffer(0, self.synapse_line_buffer.slice(..));
            let vertex_count = self.params.neuron_count * self.params.explicit_synapse_slots * 2;
            rpass.draw(0..vertex_count, 0..1);

            // 2. Draw Points (Neurons)
            rpass.set_pipeline(&self.render_pipeline_points);
            rpass.set_bind_group(0, &self.render_points_bind_group, &[]);
            rpass.set_bind_group(1, camera_bg, &[]);

            rpass.set_vertex_buffer(0, self.vertex_buffer_cube.slice(..));
            rpass.set_index_buffer(self.index_buffer_cube.slice(..), wgpu::IndexFormat::Uint16);

            // Draw 36 indices (Cube) per instance (Neuron)
            rpass.draw_indexed(0..36, 0, 0..self.params.neuron_count);
        }
    }

    pub fn get_textures(
        &self,
    ) -> (
        Arc<wgpu::Texture>,
        Arc<wgpu::TextureView>,
        Arc<wgpu::Texture>,
        Arc<wgpu::TextureView>,
    ) {
        (
            self.input_texture.clone(),
            self.input_texture_view.clone(),
            self.output_texture.clone(),
            self.output_texture_view.clone(),
        )
    }
}
