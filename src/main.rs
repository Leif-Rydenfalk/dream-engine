// src/main.rs
#![feature(portable_simd)]

use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod brain_system; // Make sure you have the brain_system.rs from the previous step
mod camera; // Make sure you have camera.rs
mod img_utils;
mod imgui_state;
mod input;
mod systems;
mod vertex;
mod wgpu_ctx; // Kept if you want to load a static image as fallback

// Re-exports
pub use app::*;
pub use brain_system::*;
pub use camera::*;
pub use imgui_state::*;
pub use input::*;
pub use systems::*;
pub use vertex::*;
pub use wgpu_ctx::*;

fn main() -> Result<(), EventLoopError> {
    env_logger::init(); // Good for debugging wgpu/cam errors
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}
