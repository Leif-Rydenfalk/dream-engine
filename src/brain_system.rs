use crate::{vertex, CameraFeed, ShaderHotReload};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const NEURON_COUNT: u32 = 1_000_000;
// A 32x32x32 grid represents the "Concept Space" (RGB Space)
// 32^3 = 32,768 buckets. Collisions will happen, which acts as stochastic noise (good for brains).
const CONCEPT_GRID_SIZE: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct NeuronGPU {
    pos: [f32; 4],    // xyz: Tissue Position (Fixed)
    weight: [f32; 4], // xyz: Semantic Position (Variable)
    state: [f32; 4],  // x: Voltage, y: Fatigue, z: Firing?, w: unused
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

pub struct BrainSystem {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    neuron_buffer: wgpu::Buffer,
    param_buffer: wgpu::Buffer,

    // The "Concept Map" - A 3D volume where neurons register their presence
    concept_map_texture: wgpu::Texture,
    concept_map_view: wgpu::TextureView,

    pub camera: CameraFeed,

    // IO Textures
    pub input_texture: Arc<wgpu::Texture>,
    pub input_texture_view: Arc<wgpu::TextureView>,
    pub output_texture: Arc<wgpu::Texture>,
    pub output_texture_view: Arc<wgpu::TextureView>,

    bind_group: wgpu::BindGroup,

    // Pipelines
    clear_pipeline: wgpu::ComputePipeline, // Step 1: Reset Grid
    populate_pipeline: wgpu::ComputePipeline, // Step 2: Neurons write to Grid
    update_pipeline: wgpu::ComputePipeline, // Step 3: Neurons read Grid & Fire
    render_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    params: SimParams,
    start_time: std::time::Instant,
}

impl BrainSystem {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_hot_reload: &ShaderHotReload,
        camera_url: String,
        surface_format: wgpu::TextureFormat, // <--- FIX: Added parameter
    ) -> Self {
        let count = NEURON_COUNT;

        // 1. Initialize Neurons
        let mut neurons = Vec::with_capacity(count as usize);
        let grid = (count as f32).powf(1.0 / 3.0).ceil() as u32;
        let spacing = 0.05;

        for i in 0..count {
            let x = (i % grid) as f32 * spacing - (grid as f32 * spacing / 2.0);
            let y = ((i / grid) % grid) as f32 * spacing - (grid as f32 * spacing / 2.0);
            let z = (i / (grid * grid)) as f32 * spacing - (grid as f32 * spacing / 2.0);

            neurons.push(NeuronGPU {
                pos: [x, y, z, 1.0],
                weight: [rand::random(), rand::random(), rand::random(), 1.0], // Random concepts initially
                state: [0.0; 4],
            });
        }

        let neuron_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brain Neurons"),
            contents: bytemuck::cast_slice(&neurons),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        });

        // 2. Concept Map (3D Grid for implicit connectivity)
        // Format R32Uint means each voxel holds ONE integer (the ID of a neuron in that bucket)
        let concept_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Concept Grid"),
            size: wgpu::Extent3d {
                width: CONCEPT_GRID_SIZE,
                height: CONCEPT_GRID_SIZE,
                depth_or_array_layers: CONCEPT_GRID_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3, // 3D Texture!
            format: wgpu::TextureFormat::R32Uint,  // Stores Neuron Index
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let concept_map_view = concept_map_texture.create_view(&Default::default());

        // 3. IO Textures
        let tex_size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };
        let input_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Eye Input"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let input_view = Arc::new(input_texture.create_view(&Default::default()));

        let output_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Brain Hallucination"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let output_view = Arc::new(output_texture.create_view(&Default::default()));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 4. Params
        let params = SimParams {
            count,
            time: 0.0,
            width: 512,
            height: 512,
            mouse_x: 0.0,
            mouse_y: 0.0,
            is_clicking: 0,
            train_mode: 1,
        };
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brain Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 5. Bind Group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Brain Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, // Neurons
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, // Params
                    visibility: wgpu::ShaderStages::COMPUTE
                        | wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, // Input Texture
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, // Sampler
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, // Output Texture
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, // CONCEPT MAP (The 3D Grid)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite, // We read and write
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brain BG"),
            layout: &bind_group_layout,
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
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&concept_map_view),
                },
            ],
        });

        // 6. Pipelines
        let shader = shader_hot_reload.get_shader("brain.wgsl");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Grid"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_clear_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        let populate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Populate Grid"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_populate_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Neurons"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_update_neurons"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Render Pipeline Setup (Cube Instancing)
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(vertex::VERTICES_CUBE),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(vertex::INDICIES_SQUARE),
            usage: wgpu::BufferUsages::INDEX,
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout, camera_bind_group_layout],
            ..Default::default()
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Brain Render"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex::create_vertex_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // FIX: Use the dynamic surface_format instead of hardcoded Rgba32Float
                targets: &[Some(surface_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            neuron_buffer,
            param_buffer,
            concept_map_texture,
            concept_map_view,
            input_texture,
            input_texture_view: input_view,
            output_texture,
            output_texture_view: output_view,
            bind_group,
            camera: CameraFeed::new(camera_url),
            clear_pipeline,
            populate_pipeline,
            update_pipeline,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: vertex::INDICIES_SQUARE.len() as u32,
            params,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn get_train_mode(&self) -> u32 {
        self.params.train_mode
    }

    pub fn update(&mut self, input: &crate::input::Input, window_size: (u32, u32)) {
        self.params.time = self.start_time.elapsed().as_secs_f32();
        let (mx, my) = input.mouse_position();
        self.params.mouse_x = (mx as f32 / window_size.0 as f32) * 2.0 - 1.0;
        self.params.mouse_y = -((my as f32 / window_size.1 as f32) * 2.0 - 1.0);
        self.params.is_clicking = if input.is_mouse_button_down(winit::event::MouseButton::Left) {
            1
        } else {
            0
        };
        self.params.train_mode = if input.is_key_down(winit::keyboard::KeyCode::Space) {
            0
        } else {
            1
        };

        if let Some(img) = self.camera.get_frame() {
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
        self.queue
            .write_buffer(&self.param_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bg: &wgpu::BindGroup,
        depth: &wgpu::TextureView,
        target: &wgpu::TextureView,
    ) {
        // 1. Clear Grid
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(4, 4, 4); // 4 * 8 = 32 covers the grid
        }

        // 2. Populate Grid (Neurons write their presence)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Populate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.populate_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (self.params.count + 63) / 64;
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        // 3. Update Neurons (Simulate and Read Grid)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (self.params.count + 63) / 64;
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        // 4. Draw
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
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_bind_group(1, camera_bg, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.num_indices, 0, 0..self.params.count);
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
