// src/camera.rs
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

pub struct CameraFeed {
    // Shared buffer between async grabber and main thread
    pub current_frame: Arc<Mutex<Option<image::RgbaImage>>>,
}

impl CameraFeed {
    pub fn new(url: String) -> Self {
        let current_frame = Arc::new(Mutex::new(None));
        let thread_buffer = current_frame.clone();

        thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                let client = reqwest::Client::new();

                loop {
                    // Reconnect loop
                    println!("Connecting to camera: {}", url);

                    // Simple approach: Repeatedly fetch the latest snapshot (snapshot.jpg)
                    // This is often easier than parsing multipart MJPEG streams manually in rust
                    // Most IP webcams expose /shot.jpg or /photo.jpg

                    match client.get(&url).send().await {
                        Ok(resp) => {
                            if let Ok(bytes) = resp.bytes().await {
                                if let Ok(img) = image::load_from_memory(&bytes) {
                                    let mut lock = thread_buffer.lock().unwrap();
                                    *lock = Some(img.to_rgba8());
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Camera error: {}", e);
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    }

                    // Cap at ~30 FPS polling
                    tokio::time::sleep(Duration::from_millis(33)).await;
                }
            });
        });

        Self { current_frame }
    }

    pub fn get_frame(&self) -> Option<image::RgbaImage> {
        let mut lock = self.current_frame.lock().unwrap();
        lock.take() // Take puts None, so we consume the frame (good for only updating on new data)
    }
}
