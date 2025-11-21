use crate::{BrainSystem, ImguiState, Input};
use cgmath::{Matrix4, SquareMatrix};
use hecs::World;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig, Texture as ImguiTexture};
use imgui_winit_support::WinitPlatform;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::window::Window;

// --- ShaderHotReload Definition ---
pub struct ShaderHotReload {
    device: Arc<wgpu::Device>,
    shader_dir: PathBuf,
    modules: Mutex<HashMap<String, (wgpu::ShaderModule, SystemTime)>>,
}

impl ShaderHotReload {
    pub fn new(device: Arc<wgpu::Device>, shader_dir: impl AsRef<Path>) -> Self {
        Self {
            device,
            shader_dir: shader_dir.as_ref().to_path_buf(),
            modules: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_shader(&self, name: &str) -> wgpu::ShaderModule {
        let path = self.shader_dir.join(name);
        let source = fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("Shader not found: {}", path.display());
        });

        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
            })
    }
}

// --- CameraUniform Definition ---
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub _padding0: u32,
    pub time: f32,
    pub _padding1: [u8; 12],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            inv_view_proj: Matrix4::identity().into(),
            view: Matrix4::identity().into(),
            camera_position: [0.0; 3],
            _padding0: 0,
            time: 0.0,
            _padding1: [0; 12],
        }
    }
}

pub struct WgpuCtx<'window> {
    surface: wgpu::Surface<'window>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Infrastructure
    shader_hot_reload: Arc<ShaderHotReload>,

    // Camera
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Rendering Targets
    depth_texture_view: wgpu::TextureView,

    // THE BRAIN
    pub brain_system: BrainSystem,

    // ImGui / UI
    pub imgui: ImguiState,
    input_texture_id: Option<TextureId>,
    output_texture_id: Option<TextureId>,

    time: Instant,
}

impl<'window> WgpuCtx<'window> {
    pub async fn new_async(window: Arc<Window>) -> Self {
        // 1. WGPU Init
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("No adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::FLOAT32_FILTERABLE
                        | wgpu::Features::VERTEX_WRITABLE_STORAGE
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    required_limits: wgpu::Limits {
                        max_compute_invocations_per_workgroup: 512,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("No Device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface_config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &surface_config);

        // 2. Utils
        let shader_hot_reload = Arc::new(ShaderHotReload::new(
            Arc::clone(&device),
            std::path::Path::new("src/shaders"),
        ));

        // 3. Camera Uniforms
        let camera_uniform = CameraUniform::default();
        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // 4. Depth Texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&Default::default());

        // 5. Initialize The Brain
        let camera_url = "http://192.168.50.208:8080/shot.jpg".to_string();

        let brain_system = BrainSystem::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &camera_bind_group_layout,
            &shader_hot_reload,
            camera_url,
            surface_config.format, // <--- FIX: Passing the format
        );

        // 6. Init ImGui
        let mut imgui = Self::init_imgui(&device, &queue, &window, surface_config.format);

        // Register Brain Textures with ImGui Renderer using from_raw_parts
        let (in_tex, in_view, out_tex, out_view) = brain_system.get_textures();
        let brain_tex_size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };

        // Config for sampler creation inside from_raw_parts
        let raw_config = imgui_wgpu::RawTextureConfig {
            label: Some("Brain Sampler"),
            sampler_desc: wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        };

        let input_tex_struct = ImguiTexture::from_raw_parts(
            &device,
            &imgui.renderer,
            in_tex,
            in_view,
            None,
            Some(&raw_config),
            brain_tex_size,
        );

        let output_tex_struct = ImguiTexture::from_raw_parts(
            &device,
            &imgui.renderer,
            out_tex,
            out_view,
            None,
            Some(&raw_config),
            brain_tex_size,
        );

        let input_texture_id = Some(imgui.renderer.textures.insert(input_tex_struct));
        let output_texture_id = Some(imgui.renderer.textures.insert(output_tex_struct));

        Self {
            surface,
            surface_config,
            device,
            queue,
            shader_hot_reload,
            camera_buffer,
            camera_bind_group,
            depth_texture_view,
            brain_system,
            imgui,
            input_texture_id,
            output_texture_id,
            time: Instant::now(),
        }
    }

    pub fn draw(&mut self, _world: &mut World, window: &Window, input: &Input) {
        // 1. Update CPU State
        let window_size = (self.surface_config.width, self.surface_config.height);
        self.brain_system.update(input, window_size);

        // 2. Prepare Render Pass
        let surface_texture = self.surface.get_current_texture().unwrap();
        let view = surface_texture.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // 3. Render The Brain (Compute + 3D Draw)
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }

        self.brain_system.render(
            &mut encoder,
            &self.camera_bind_group,
            &self.depth_texture_view,
            &view,
        );

        // 4. Render UI (Camera Feed & Hallucinations)
        self.render_imgui(&mut encoder, &view, window);

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }

    fn render_imgui(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        window: &Window,
    ) {
        let now = Instant::now();
        self.imgui
            .context
            .io_mut()
            .update_delta_time(now - self.imgui.last_frame);
        self.imgui.last_frame = now;

        {
            let ui = self.imgui.context.frame();

            // --- DASHBOARD WINDOW ---
            ui.window("Neural Interlink")
                .position([10.0, 10.0], Condition::FirstUseEver)
                .size([600.0, 400.0], Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("FPS: {:.1}", ui.io().framerate));
                    ui.separator();

                    // Show Training Mode status
                    let mode_text = if self.brain_system.get_train_mode() == 1 {
                        "Mode: TRAINING (Input On)"
                    } else {
                        "Mode: DREAMING (Input Off)"
                    };
                    ui.text(mode_text);
                    ui.text("Hold SPACE to toggle Dream Mode");
                    ui.separator();

                    // Show Inputs/Outputs Side by Side
                    if let (Some(in_id), Some(out_id)) =
                        (self.input_texture_id, self.output_texture_id)
                    {
                        ui.columns(2, "cam_cols", true);

                        ui.text("Optic Nerve (Input)");
                        Image::new(in_id, [256.0, 256.0]).build(ui);

                        ui.next_column();

                        ui.text("Visual Cortex (Activity)");
                        Image::new(out_id, [256.0, 256.0]).build(ui);

                        ui.columns(1, "reset", false);
                    }

                    ui.separator();
                    ui.text("Controls:");
                    ui.text("- Mouse Wheel: Zoom");
                    ui.text("- Left Click: Electrical Stimulus");
                    ui.text("- W/A/S/D: Move Camera");
                });

            // Prepare the platform (handles cursor, etc.)
            self.imgui.platform.prepare_render(&ui, window);
        }

        let draw_data = self.imgui.context.render();

        self.imgui
            .renderer
            .render(
                draw_data,
                &self.queue,
                &self.device,
                &mut encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Imgui Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                }),
            )
            .expect("Imgui Render Error");
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.surface_config.width = new_size.0.max(1);
        self.surface_config.height = new_size.1.max(1);
        self.surface.configure(&self.device, &self.surface_config);

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
            size: wgpu::Extent3d {
                width: new_size.0,
                height: new_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_texture_view = depth_texture.create_view(&Default::default());
    }

    pub fn update_camera_uniform(
        &mut self,
        view_proj: Matrix4<f32>,
        inv: Matrix4<f32>,
        view: Matrix4<f32>,
        pos: [f32; 3],
    ) {
        let u = CameraUniform {
            view_proj: view_proj.into(),
            inv_view_proj: inv.into(),
            view: view.into(),
            camera_position: pos,
            time: self.time.elapsed().as_secs_f32(),
            ..Default::default()
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[u]));
    }

    fn init_imgui(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Window,
        format: wgpu::TextureFormat,
    ) -> ImguiState {
        let mut context = imgui::Context::create();
        let mut platform = WinitPlatform::new(&mut context);
        platform.attach_window(
            context.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );
        context.set_ini_filename(None);

        let renderer_config = RendererConfig {
            texture_format: format,
            ..Default::default()
        };
        let renderer = Renderer::new(&mut context, device, queue, renderer_config);

        ImguiState {
            context,
            platform,
            renderer,
            clear_color: wgpu::Color::BLACK,
            demo_open: false,
            last_frame: Instant::now(),
            last_cursor: None,
        }
    }
}
