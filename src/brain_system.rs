use crate::{vertex, CameraFeed, ShaderHotReload};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const NEURON_COUNT: u32 = 1_000_000;
const GRID_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct NeuronGPU {
    pos_physical: [f32; 4],
    pos_semantic: [f32; 4],
    state: [f32; 4],
    velocity: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SimParams {
    count: u32,
    time: f32,
    width: u32,
    height: u32,
    train_mode: u32,
    _padding1: f32, // Padding to align to 16 bytes
    _padding2: f32,
    _padding3: f32,
}

pub struct BrainSystem {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    neuron_buffer: wgpu::Buffer,
    param_buffer: wgpu::Buffer,
    spatial_grid_buffer: wgpu::Buffer,
    pub camera: CameraFeed,
    pub input_texture: Arc<wgpu::Texture>,
    pub input_texture_view: Arc<wgpu::TextureView>,
    pub output_texture: Arc<wgpu::Texture>,
    pub output_texture_view: Arc<wgpu::TextureView>,
    compute_bind_group: wgpu::BindGroup,
    empty_bind_group: wgpu::BindGroup,
    clear_grid_pipeline: wgpu::ComputePipeline,
    populate_grid_pipeline: wgpu::ComputePipeline,
    update_neurons_pipeline: wgpu::ComputePipeline,
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
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let count = NEURON_COUNT;

        let mut neurons = Vec::with_capacity(count as usize);
        let grid_dim_phys = (count as f32).powf(0.5).ceil() as usize;

        for i in 0..count {
            let u = (i % grid_dim_phys as u32) as f32 / grid_dim_phys as f32;
            let v = (i / grid_dim_phys as u32) as f32 / grid_dim_phys as f32;
            let px = u * 2.0 - 1.0;
            let py = v * 2.0 - 1.0;

            let sx = (rand::random::<f32>() * 2.0 - 1.0) * 3.0; // Spread out start
            let sy = (rand::random::<f32>() * 2.0 - 1.0) * 3.0;
            let sz = (rand::random::<f32>() * 2.0 - 1.0) * 3.0;

            neurons.push(NeuronGPU {
                pos_physical: [px, py, 0.0, 0.0],
                pos_semantic: [sx, sy, sz, 1.0],
                state: [0.0, 0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0, 0.0],
            });
        }

        let neuron_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brain Neurons"),
            contents: bytemuck::cast_slice(&neurons),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        });

        let grid_len = GRID_SIZE * GRID_SIZE * GRID_SIZE;
        let spatial_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Grid Buffer"),
            size: (grid_len as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let input_view = Arc::new(input_texture.create_view(&Default::default()));

        let output_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Visual Cortex Output"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let output_view = Arc::new(output_texture.create_view(&Default::default()));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params = SimParams {
            count,
            time: 0.0,
            width: 512,
            height: 512,
            train_mode: 1,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
        };
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brain Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Brain Compute Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
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
                ],
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brain Compute BG"),
            layout: &compute_bind_group_layout,
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
                    resource: spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });

        let empty_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Empty Layout"),
                entries: &[],
            });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Empty BG"),
            layout: &empty_bind_group_layout,
            entries: &[],
        });

        let shader = shader_hot_reload.get_shader("brain.wgsl");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let clear_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Grid"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_clear_grid"),
                compilation_options: Default::default(),
                cache: None,
            });

        let populate_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Populate Grid"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_populate_grid"),
                compilation_options: Default::default(),
                cache: None,
            });

        let update_neurons_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update Neurons"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_update_neurons"),
                compilation_options: Default::default(),
                cache: None,
            });

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
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&empty_bind_group_layout, camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<NeuronGPU>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Brain Render"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex::create_vertex_buffer_layout(), instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            spatial_grid_buffer,
            input_texture,
            input_texture_view: input_view,
            output_texture,
            output_texture_view: output_view,
            compute_bind_group,
            empty_bind_group,
            camera: CameraFeed::new(camera_url),
            clear_grid_pipeline,
            populate_grid_pipeline,
            update_neurons_pipeline,
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

    pub fn update(&mut self, input: &crate::input::Input, _window_size: (u32, u32)) {
        self.params.time = self.start_time.elapsed().as_secs_f32();
        // REMOVED MOUSE INPUT LOGIC HERE
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
                label: Some("Clear Grid"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_grid_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            let grid_num = GRID_SIZE * GRID_SIZE * GRID_SIZE;
            cpass.dispatch_workgroups((grid_num + 63) / 64, 1, 1);
        }
        // 2. Populate Grid
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Populate Grid"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.populate_grid_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.count + 63) / 64, 1, 1);
        }
        // 3. Update Neurons
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update Neurons"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_neurons_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.params.count + 63) / 64, 1, 1);
        }
        // 4. Render 3D
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
            rpass.set_bind_group(0, &self.empty_bind_group, &[]);
            rpass.set_bind_group(1, camera_bg, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, self.neuron_buffer.slice(..));
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
