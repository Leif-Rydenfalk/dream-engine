// src/app.rs
use cgmath::{Point3, SquareMatrix};
use hecs::World;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::MouseScrollDelta::*;
use winit::event::WindowEvent;
use winit::event::{DeviceEvent, DeviceId};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::input::Input;
use crate::wgpu_ctx::WgpuCtx;
use crate::*;

#[derive(Default)]
pub struct App<'window> {
    window: Option<Arc<Window>>,
    wgpu_ctx: Option<WgpuCtx<'window>>,
    input_system: Input,
    world: World,
    camera_entity: Option<hecs::Entity>,
    last_frame_time: Option<Instant>,
}

impl<'window> ApplicationHandler for App<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes()
                .with_title("Neural Chassis")
                .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
            let window = Arc::new(event_loop.create_window(win_attr).unwrap());

            // Async creation wrapper
            let wgpu_ctx = pollster::block_on(WgpuCtx::new_async(window.clone()));

            self.window = Some(window.clone());
            self.wgpu_ctx = Some(wgpu_ctx);
            self.world = World::new();

            // Setup Camera
            let window_size = window.inner_size();
            self.camera_entity = Some(setup_camera_entity(
                &mut self.world,
                Some((window_size.width, window_size.height)),
            ));

            // Adjust camera start position to see the brain center
            if let Some(e) = self.camera_entity {
                if let Ok(t) = self.world.query_one_mut::<&mut Transform>(e) {
                    t.position = Point3::new(0.0, 0.0, -12.0); // Look at 0,0,0
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event.clone() {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(wgpu_ctx), Some(_)) = (self.wgpu_ctx.as_mut(), self.window.as_ref()) {
                    wgpu_ctx.resize((new_size.width, new_size.height));
                    if let Some(entity) = self.camera_entity {
                        if let Ok(camera) = self.world.query_one_mut::<&mut Camera>(entity) {
                            camera.aspect = new_size.width as f32 / new_size.height as f32;
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame_time
                    .map(|t| now.duration_since(t))
                    .unwrap_or_default();
                self.last_frame_time = Some(now);

                update_camera_system(&mut self.world, &self.input_system, dt);

                // Update Camera Uniforms
                if let (Some(wgpu_ctx), Some(entity)) = (&mut self.wgpu_ctx, self.camera_entity) {
                    if let Ok((t, c)) = self.world.query_one_mut::<(&Transform, &Camera)>(entity) {
                        let view_proj = calculate_view_projection(t, c);
                        let inv = view_proj.invert().unwrap();
                        let view = calculate_view_matrix(t);
                        wgpu_ctx.update_camera_uniform(view_proj, inv, view, t.position.into());
                    }

                    // DRAW
                    wgpu_ctx.draw(
                        &mut self.world,
                        self.window.as_mut().unwrap(),
                        &self.input_system,
                    );
                }

                self.input_system.update();
                self.window.as_ref().unwrap().request_redraw(); // Continuous loop
            }
            // Input Handling (Pass to ImGui + Input System)
            WindowEvent::KeyboardInput { event, .. } => {
                let io = self.wgpu_ctx.as_mut().unwrap().imgui.context.io();
                if !io.want_capture_keyboard {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        event_loop.exit();
                    }
                    if let PhysicalKey::Code(key) = event.physical_key {
                        self.input_system.handle_key_input(key, event.state);
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if !self
                    .wgpu_ctx
                    .as_mut()
                    .unwrap()
                    .imgui
                    .context
                    .io()
                    .want_capture_mouse
                {
                    self.input_system.handle_mouse_button(button, state);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input_system.handle_cursor_moved(&position);
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                LineDelta(_, y) => self.input_system.handle_mouse_scroll(y as f64),
                PixelDelta(d) => self.input_system.handle_mouse_scroll(d.y),
            },
            _ => (),
        }

        // Pass to ImGui
        if let Some(ctx) = self.wgpu_ctx.as_mut() {
            ctx.imgui.platform.handle_event::<()>(
                ctx.imgui.context.io_mut(),
                self.window.as_ref().unwrap(),
                &winit::event::Event::WindowEvent { window_id, event },
            );
        }
    }
}

pub fn setup_camera_entity(world: &mut World, window_size: Option<(u32, u32)>) -> hecs::Entity {
    // Calculate initial aspect ratio based on window size, or use default if not provided
    let aspect = if let Some((width, height)) = window_size {
        width as f32 / height as f32
    } else {
        16.0 / 9.0 // Default aspect ratio
    };

    world.spawn((
        Transform {
            position: Point3::new(0.0, 1.0, 3.0),
            ..Default::default()
        },
        Camera {
            aspect,
            ..Default::default()
        },
        CameraController::default(),
    ))
}
