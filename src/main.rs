use std::borrow::Cow;
use std::{env, io};
use std::io::Write;
use imgui_wgpu::{Renderer, RendererConfig};
use pollster::block_on;

use std::time::Instant;
use ansi_term::Color::{Blue, Green, Purple, Red, Yellow};
use ansi_term::Style;
use env_logger::{Builder, Target};
use glam::{Mat4, Vec2, Vec3};
use imgui::{Condition, Context, FontSource};
use imgui_winit_support::WinitPlatform;
use log::{info, LevelFilter};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use wgpu::util::DeviceExt;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
    window::Window,
};
use winit::dpi::Position;

const CHILD_OFFSET: u32 = 24;
const SVO_DEPTH: u8 = 8;

pub trait Octant {
    fn set_child(&self, child: u32) -> u32;
    fn check_child(&self, child: u32) -> bool;
    fn has_children(&self) -> bool;
    fn set_first_child_index(&self, index: u32) -> u32;
    fn first_child_index(&self) -> u32;
    fn child_mask(&self) -> u8;
    fn set_child_mask(&self, mask: u8) -> u32;
    fn child_count(&self) -> u32;
}

impl Octant for u32 {
    fn set_child(&self, child: u32) -> u32 {
        self | 1u32 << (child + CHILD_OFFSET)
    }

    fn check_child(&self, child: u32) -> bool {
        self & (1u32 << (child + CHILD_OFFSET)) > 0
    }

    fn has_children(&self) -> bool {
        self & 0b11111111_00000000_00000000_00000000 > 0
    }

    fn set_first_child_index(&self, index: u32) -> u32 {
        (self & 0b11111111_00000000_00000000_00000000) | (index & 0b00000000_11111111_11111111_11111111)
    }

    fn first_child_index(&self) -> u32 {
        self & 0b00000000_11111111_11111111_11111111
    }

    fn child_mask(&self) -> u8 {
        ((self & 0b11111111_00000000_00000000_00000000) >> CHILD_OFFSET) as u8
    }

    fn set_child_mask(&self, mask: u8) -> u32 {
        (self & 0b00000000_11111111_11111111_11111111) | (((mask as u32) << CHILD_OFFSET) & 0b11111111_00000000_00000000_00000000)
    }

    fn child_count(&self) -> u32 {
        (self >> CHILD_OFFSET).count_ones()
    }
}

pub fn encode_node(child_mask: u8, first_child_index: u32) -> u32 {
    ((child_mask as u32) << CHILD_OFFSET) | (first_child_index & 0b00000000_11111111_11111111_11111111)
}


struct SVO {
    nodes: Vec<u32>,
    depth: u8,
}

impl SVO {
    fn new(depth: u8) -> Self {
        SVO {
            // insert root
            nodes: Vec::from([0]),
            depth,
        }
    }

    fn gen_random_svo(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        self.gen_random_branch(&mut rng, 0, 0); // start at root index & depth
    }

    fn gen_random_branch(&mut self, rng: &mut StdRng, cur_index: usize, cur_depth: u8) {
        if cur_depth < self.depth {
            let child_mask = rng.gen::<u8>();
            let first_child_index = self.nodes.len() as u32;
            let node = encode_node(child_mask, first_child_index);
            self.nodes[cur_index] = node;

            for i in 0..8 {
                self.nodes.push(0);

                if node.check_child(i) {
                    self.gen_random_branch(rng, (first_child_index + i) as usize, cur_depth + 1);
                }
            }
        }

        return;
    }

    fn count_notes(&self) -> u32 {
        self.nodes.iter().map(|&n| n > 1).count() as u32
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    proj_mat: [[f32; 4]; 4],
    time: u32,
    _padding: [u32; 3],
}

impl Default for Uniform {
    fn default() -> Self {
        Uniform {
            proj_mat: Mat4::IDENTITY.to_cols_array_2d(),
            time: 0,
            _padding: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0, 0.0], uv: [0.0, 0.0] },
    Vertex { position: [1.0, -1.0, 0.0], uv: [1.0, 0.0] },
    Vertex { position: [1.0, 1.0, 0.0], uv: [1.0, 1.0] },
    Vertex { position: [-1.0, 1.0, 0.0], uv: [0.0, 1.0] },
];

const INDICES: &[u16] = &[
    0, 1, 2,
    2, 3, 0,
];

struct Camera {
    // settings
    mov_speed_slow: f32,
    mov_speed_fast: f32,
    
    fov: f32,
    sensitivity: f32,

    z_near: f32,
    z_far: f32,

    rotation_min: Vec2, // x: yaw, y: pitch
    rotation_max: Vec2, // x: yaw, y: pitch

    up_axis: Vec3,

    enable_matrices: bool,

    // internal
    rotation: Vec2, // x: yaw, y: pitch

    front: Vec3,
    right: Vec3,
    up: Vec3,

    position: Vec3,

    mov_speed: f32,
    mov_lin: Vec3,
    mov_lat: Vec3,

    plane_u: Vec3,
    plane_v: Vec3,

    aspect_ratio: f32,
    fov_tan: f32,

    proj: Mat4,
}

impl Camera {
    fn update_proj(&mut self, window_width: u32, window_height: u32) {
        self.aspect_ratio = (window_width as f32) / (window_height as f32);
        if self.enable_matrices {
            self.proj = Mat4::perspective_lh(self.fov_tan, self.aspect_ratio, self.z_near, self.z_far);
        }
    }

    fn update_camera_rotation(&mut self, mouse_delta: Vec2) {
        self.rotation -= mouse_delta * self.sensitivity;
        self.rotation = self.rotation.clamp(self.rotation_min, self.rotation_max);

        self.front = Vec3::new(
            self.rotation.x.to_radians().cos() * self.rotation.y.to_radians().cos(),
            self.rotation.y.to_radians().sin(),
            self.rotation.x.to_radians().sin() * self.rotation.y.to_radians().cos(),
        ).normalize();

        self.right = self.front.cross(self.up_axis).normalize();
        self.up = self.right.cross(self.front).normalize();

        self.mov_lin = self.front * self.mov_speed;
        self.mov_lat = self.right * self.mov_speed;

        self.plane_u = self.right * self.aspect_ratio * self.fov_tan;
        self.plane_v = self.up * self.aspect_ratio * self.fov_tan;
    }

    fn get_view_proj(&self) -> Mat4 {
        if self.enable_matrices {
            let view = Mat4::look_at_lh(self.position, self.position + self.front, self.up);
            self.proj * view
        } else {
            Mat4::IDENTITY
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            // settings
            mov_speed_slow: 5.0,
            mov_speed_fast: 10.0,
            
            fov: 60.0,
            sensitivity: 0.1,

            z_near: 0.1,
            z_far: 100.0,

            rotation_min: Vec2::new(-180.0, -89.0),
            rotation_max: Vec2::new(180.0, 89.0),

            up_axis: Vec3::new(0.0, 1.0, 0.0),

            enable_matrices: false,

            // internal
            rotation: Vec2::ZERO,

            front: Vec3::ZERO,
            right: Vec3::ZERO,
            up: Vec3::ZERO,

            position: Vec3::ZERO,

            mov_speed: 0.0,
            mov_lin: Vec3::ZERO,
            mov_lat: Vec3::ZERO,

            plane_u: Vec3::ZERO,
            plane_v: Vec3::ZERO,

            aspect_ratio: 0.0,
            fov_tan: 0.0,

            proj: Mat4::IDENTITY,
        }
    }
}

async fn run(event_loop: EventLoop<()>, window: Window, svo: SVO) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: wgpu::InstanceFlags::debugging(),
        ..Default::default()
    });

    let surf = instance.create_surface(&window).unwrap();

    let hidpi_factor = window.scale_factor();

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surf),
        force_fallback_adapter: false,
    })).unwrap();

    let (dev, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 1,
                    max_storage_buffer_binding_size:  !(!0 << CHILD_OFFSET),
                    
                    ..wgpu::Limits::downlevel_webgl2_defaults()
                }
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("failed to create device.");
    //
    // create buffers
    //
    let mut uniform: Uniform = Default::default();

    let uniform_buffer = dev.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }
    );

    // TODO: add vertex staging buffer
    let num_vertices = VERTICES.len() as u32;
    let vertex_buffer = dev.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }
    );

    // TODO: add index staging buffer
    let num_indices = INDICES.len() as u32;
    let index_buffer = dev.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        }
    );

    // TODO: add svo staging buffer
    let svo_buffer = dev.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("svo_buffer"),
            contents: bytemuck::cast_slice(&svo.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        }
    );
    //
    // bind group
    //
    let bind_group_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("bind_group_layout"),
    });

    let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: svo_buffer.as_entire_binding(),
            },
        ],
        label: Some("bind_group"),
    });
    //
    // create pipeline
    //
    let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let pipeline_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[
            &bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surf.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[Vertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut surf_cfg = surf
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();

    surf.configure(&dev, &surf_cfg);
    //
    // imgui setup
    //
    let mut imgui = Context::create();
    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);
    imgui.set_ini_filename(None);

    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(imgui::FontConfig {
            oversample_h: 1,
            pixel_snap_h: true,
            size_pixels: font_size,
            ..Default::default()
        }),
    }]);

    let renderer_cfg = RendererConfig {
        texture_format: surf_cfg.format,
        ..Default::default()
    };

    let mut renderer = Renderer::new(&mut imgui, &dev, &queue, renderer_cfg);

    //
    // main loop, window event handling
    //
    let mut last_frame = Instant::now();
    let mut last_cursor = None;

    let clear_color = wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 };

    event_loop
        .run(|event, elwt| {
            let _ = (&instance, &adapter, &shader, &pipeline_layout);
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::AboutToWait => window.request_redraw(),
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(new_size) => {
                        surf_cfg.width = new_size.width.max(1);
                        surf_cfg.height = new_size.height.max(1);
                        surf.configure(&dev, &surf_cfg);
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let Key::Named(NamedKey::Escape) = event.logical_key {
                            if event.state.is_pressed() {
                                elwt.exit();
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let delta_time = last_frame.elapsed();
                        imgui.io_mut().update_delta_time(delta_time);
                        last_frame = Instant::now();

                        let frame = surf
                            .get_current_texture()
                            .expect("failed to acquire next swapchain texture.");

                        uniform.time += 1;
                        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

                        platform
                            .prepare_frame(imgui.io_mut(), &window)
                            .expect("failed to prepare frame.");

                        let ui = imgui.frame();

                        {
                            ui.window("info")
                                .size([400.0, 200.0], Condition::FirstUseEver)
                                .position([400.0, 200.0], Condition::FirstUseEver)
                                .build(|| {
                                    ui.text(format!("frame_time: {delta_time:?}"));
                                    let mouse_pos = ui.io().mouse_pos;
                                    ui.text(format!(
                                        "mouse_pos: ({:.1},{:.1})",
                                        mouse_pos[0], mouse_pos[1]
                                    ));
                                });
                        }

                        let mut encoder: wgpu::CommandEncoder = dev
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        if last_cursor != Some(ui.mouse_cursor()) {
                            last_cursor = Some(ui.mouse_cursor());
                            platform.prepare_render(ui, &window);
                        }

                        {
                            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(clear_color),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });

                            render_pass.set_pipeline(&render_pipeline);
                            render_pass.set_bind_group(0, &bind_group, &[]);
                            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                            render_pass.draw_indexed(0..num_indices, 0, 0..1);

                            renderer
                                .render(imgui.render(), &queue, &dev, &mut render_pass)
                                .expect("rendering imgui failed.");
                        }

                        queue.submit(Some(encoder.finish()));
                        frame.present();
                    }
                    _ => {}
                },
                _ => {}
            }

            platform.handle_event(imgui.io_mut(), &window, &event);
        }).unwrap();
}

fn main() {
    env::set_var("RUST_LOG", "info");
    let mut builder = Builder::from_default_env();

    builder
        .target(Target::Stdout)
        .format(|buf, record| {
            let level = record.level();
            let style = match level {
                log::Level::Error => Style::new().bold().fg(Red),
                log::Level::Warn => Style::new().bold().fg(Yellow),
                log::Level::Info => Style::new().bold().fg(Blue),
                //log::Level::Debug => Style::new().fg(Purple),
                //log::Level::Trace => Style::new().fg(Green),
                _ => return Ok(()),
            };

            buf.write_fmt(format_args!(
                "{}: {}\n",
                style.paint(record.level().to_string()),
                record.args()
            ))
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "Error writing log"))
        })
        .filter(None, LevelFilter::Trace) // Adjust the filter to show all levels
        .init();

    println!();
    info!("logger initialized.");

    let event_loop = EventLoop::new().unwrap();

    let window = {
        let size = LogicalSize::new(1280.0, 720.0);
        WindowBuilder::new()
            .with_inner_size(size)
            .with_title(env!("CARGO_PKG_NAME"))
            .build(&event_loop)
            .unwrap()
    };

    let mut svo = SVO::new(SVO_DEPTH);
    svo.gen_random_svo(0);

    info!("filled node count: {}", svo.count_notes());

    block_on(run(event_loop, window, svo));
}