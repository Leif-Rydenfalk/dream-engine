use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod brain_system;
mod camera;
mod img_utils;
mod imgui_state;
mod input;
mod model;
mod systems;
mod vertex;
mod wgpu_ctx;

// Re-exports
pub use app::*;
pub use brain_system::*;
pub use camera::*;
pub use imgui_state::*;
pub use input::*;
pub use model::*;
pub use systems::*;
pub use vertex::*;
pub use wgpu_ctx::*;

fn main() -> Result<(), EventLoopError> {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)
}
