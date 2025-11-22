use crate::{vertex, CameraFeed, ShaderHotReload};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// OPTIMIZATION: Reduced neuron count and retina size
// 160*160 = 25,600 Retina neurons
// 200,000 - 25,600 = 174,400 Cortex neurons
const NEURON_COUNT: u32 = 200_000;
const GRID_DIM: u32 = 128;
// OPTIMIZATION: Reduced explicit slots per neuron from 32 -> 16
const EXPLICIT_SLOTS: usize = 16;

// --- GPU STRUCTS ---

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Neuron {
    // BLOCK 1 (16 bytes)
    pub concept_pos: [f32; 4],

    // BLOCK 2 (16 bytes)
    pub cortical_pos: [f32; 2],
    pub retinal_coord: [u32; 2],

    // BLOCK 3 (16 bytes)
    pub layer: u32,
    pub voltage: f32,
    pub plasticity_trace: f32,
    pub learning_rate: f32,

    // BLOCK 4 (16 bytes)
    pub surprise_accumulator: f32,
    pub top_geometric_contrib: f32,
    pub top_geometric_source: u32,
    pub _pad0: f32,

    // EXPLICIT SYNAPSES (Aligned 16-byte boundaries)
    // Fixed 16 slots = 16 * 4 = 64 bytes per array
    pub explicit_targets: [u32; EXPLICIT_SLOTS],
    pub explicit_weights: [f32; EXPLICIT_SLOTS],
    pub explicit_ages: [f32; EXPLICIT_SLOTS],
    pub explicit_visual_weights: [f32; EXPLICIT_SLOTS],
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
    pub use_camera: u32,
    pub _pad: [f32; 7],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LineVertex {
    pos: [f32; 4],
    color: [f32; 4],
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
    pub prediction_texture: Arc<wgpu::Texture>,
    pub prediction_texture_view: Arc<wgpu::TextureView>,

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
    clear_cortex_pipeline: wgpu::ComputePipeline,
    render_dream_pipeline: wgpu::ComputePipeline,

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
            geometric_sample_count: 16, // Reduced from 64
            explicit_synapse_slots: EXPLICIT_SLOTS as u32,
            train_mode: 1,
            terror_threshold: 0.5,
            grid_dim: GRID_DIM,
            use_camera: 1,
            _pad: [0.0; 7],
        };

        let kernel = SynapticKernel {
            local_amplitude: 2.5,
            local_decay: 0.05,
            conceptual_amplitude: 1.2,
            conceptual_decay: 1.5,
            inhibit_radius: 0.15,
            inhibit_strength: 0.8,
            explicit_learning_rate: 0.15,
            pruning_threshold: 5.0,
            promotion_threshold: 0.3,
            temporal_decay: 0.01,
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

        let spike_history_len = NEURON_COUNT * 4; // Reduced history
        let spike_history_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spike History"),
            size: (spike_history_len as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let max_lines = NEURON_COUNT as usize * EXPLICIT_SLOTS;
        let line_buffer_size = (max_lines * std::mem::size_of::<LineVertex>() * 2) as u64;
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

        // 3. Textures
        let tex_size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };

        let input_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input Tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let input_texture_view = Arc::new(input_texture.create_view(&Default::default()));

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

        let prediction_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prediction Tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let prediction_texture_view = Arc::new(prediction_texture.create_view(&Default::default()));

        let input_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Input Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 4. Bind Groups
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Brain Compute Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
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
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&prediction_texture_view),
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

        let shader_module = shader_hot_reload.get_shader("brain.wgsl");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[&compute_layout],
            push_constant_ranges: &[],
        });

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
        let clear_cortex_pipeline = create_compute("cs_clear_cortex");
        let render_dream_pipeline = create_compute("cs_render_dream");

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
                                format: wgpu::VertexFormat::Float32x4, // Position
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4, // Color
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
            prediction_texture,
            prediction_texture_view,
            camera: CameraFeed::new(camera_url),
            compute_bind_group,
            render_points_bind_group,
            empty_bind_group,
            init_pipeline,
            clear_grid_pipeline,
            populate_grid_pipeline,
            update_neurons_pipeline,
            generate_lines_pipeline,
            clear_cortex_pipeline,
            render_dream_pipeline,
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

        // Toggle Train Mode (Space)
        if input.is_key_pressed(winit::keyboard::KeyCode::Space) {
            self.params.train_mode = if self.params.train_mode == 1 { 0 } else { 1 };
        }

        // Toggle Dream Mode (D)
        if input.is_key_pressed(winit::keyboard::KeyCode::KeyD) {
            self.params.use_camera = if self.params.use_camera == 1 { 0 } else { 1 };
            println!(
                "Input Source: {}",
                if self.params.use_camera == 1 {
                    "CAMERA"
                } else {
                    "DREAM (Feedback Loop)"
                }
            );
        }

        // Update Camera Input ONLY if active
        if self.params.use_camera == 1 {
            if let Some(img) = self.camera.get_frame() {
                let img_resized =
                    image::imageops::resize(&img, 512, 512, image::imageops::FilterType::Nearest);
                let raw_f32: Vec<f32> = img_resized
                    .as_raw()
                    .iter()
                    .map(|&b| b as f32 / 255.0)
                    .collect();
                let raw_bytes = bytemuck::cast_slice(&raw_f32);

                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.input_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    raw_bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(16 * 512),
                        rows_per_image: Some(512),
                    },
                    wgpu::Extent3d {
                        width: 512,
                        height: 512,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

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
        // DREAM MODE: Feed Prediction back into Input
        if self.params.use_camera == 0 {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.prediction_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: 512,
                    height: 512,
                    depth_or_array_layers: 1,
                },
            );
        }

        const RENDER_3D_VISUALIZATION: bool = true;

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

            // 2. Populate Grid
            cpass.set_pipeline(&self.populate_grid_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);

            // 3. Clear Visualization Textures (BEFORE neurons write to them)
            cpass.set_pipeline(&self.clear_cortex_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(512 / 8, 512 / 8, 1);

            cpass.set_pipeline(&self.render_dream_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(512 / 8, 512 / 8, 1);

            // 4. Update Neurons (which write to the cortex texture)
            cpass.set_pipeline(&self.update_neurons_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);

            // 5. Generate Lines
            if RENDER_3D_VISUALIZATION {
                cpass.set_pipeline(&self.generate_lines_pipeline);
                cpass.set_bind_group(0, &self.compute_bind_group, &[]);
                cpass.dispatch_workgroups((self.params.neuron_count + 63) / 64, 1, 1);
            }
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Brain Draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

            if RENDER_3D_VISUALIZATION {
                rpass.set_pipeline(&self.render_pipeline_lines);
                rpass.set_bind_group(0, &self.empty_bind_group, &[]);
                rpass.set_bind_group(1, camera_bg, &[]);
                rpass.set_vertex_buffer(0, self.synapse_line_buffer.slice(..));
                let vertex_count =
                    self.params.neuron_count * self.params.explicit_synapse_slots * 2;
                rpass.draw(0..vertex_count, 0..1);

                rpass.set_pipeline(&self.render_pipeline_points);
                rpass.set_bind_group(0, &self.render_points_bind_group, &[]);
                rpass.set_bind_group(1, camera_bg, &[]);

                rpass.set_vertex_buffer(0, self.vertex_buffer_cube.slice(..));
                rpass.set_index_buffer(self.index_buffer_cube.slice(..), wgpu::IndexFormat::Uint16);

                rpass.draw_indexed(0..36, 0, 0..self.params.neuron_count);
            }
        }
    }

    pub fn get_textures(
        &self,
    ) -> (
        Arc<wgpu::Texture>,
        Arc<wgpu::TextureView>,
        Arc<wgpu::Texture>,
        Arc<wgpu::TextureView>,
        Arc<wgpu::Texture>,
        Arc<wgpu::TextureView>,
    ) {
        (
            self.input_texture.clone(),
            self.input_texture_view.clone(),
            self.prediction_texture.clone(),
            self.prediction_texture_view.clone(),
            self.output_texture.clone(),
            self.output_texture_view.clone(),
        )
    }
}
