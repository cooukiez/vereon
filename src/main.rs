mod camera;
mod octree;
mod types;

use imgui_wgpu::{Renderer, RendererConfig};
use pollster::block_on;
use std::borrow::Cow;
use std::io::Write;
use std::{env, io};

use crate::camera::Camera;
use crate::octree::SVO;
use crate::types::{Uniform, Vertex, INDICES, VERTICES};
use ansi_term::Color::{Blue, Red, Yellow};
use ansi_term::Style;
use env_logger::{Builder, Target};
use glam::{UVec2, Vec3};
use glam::Vec2;
use imgui::{Condition, Context, FontSource};
use imgui_winit_support::WinitPlatform;
use log::{info, LevelFilter};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalPosition;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::Window,
    window::WindowBuilder,
};
use winit::event::ElementState;
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

const CHILD_OFFSET: u32 = 24;
const SVO_DEPTH: u8 = 8;

async fn run(event_loop: EventLoop<()>, window: Window, svo: SVO) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::debugging(),
        ..Default::default()
    });

    let surf = instance.create_surface(&window).unwrap();

    let hidpi_factor = window.scale_factor();

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surf),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (dev, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 1,
                    max_storage_buffer_binding_size: !(!0 << CHILD_OFFSET),

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
    let mut cam = Camera::default();
    cam.update_proj(size.width, size.height);

    let mut uniform: Uniform = Uniform {
        cam_pos: cam.pos.extend(0.0).to_array(),
        cam_dir: cam.front.extend(0.0).to_array(),
        cam_plane_u: cam.plane_u.extend(0.0).to_array(),
        cam_plane_v: cam.plane_v.extend(0.0).to_array(),
        res: UVec2::new(size.width, size.height).to_array(),
        ..Default::default()
    };

    let uniform_buffer = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // TODO: add vertex staging buffer
    let vertex_buffer = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex_buffer"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // TODO: add index staging buffer
    let num_indices = INDICES.len() as u32;
    let index_buffer = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("index_buffer"),
        contents: bytemuck::cast_slice(INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    // TODO: add svo staging buffer
    let svo_buffer = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("svo_buffer"),
        contents: bytemuck::cast_slice(&svo.nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });
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
    /*
    let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/shader.wgsl"))),
    });
    */
    
    let vert_shader = unsafe { dev.create_shader_module_spirv(&wgpu::include_spirv_raw!("shader/glsl/vert.spv")) };
    let frag_shader = unsafe { dev.create_shader_module_spirv(&wgpu::include_spirv_raw!("shader/glsl/frag.spv")) };

    let pipeline_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surf.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vert_shader,
            entry_point: "main",
            buffers: &[Vertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag_shader,
            entry_point: "main",
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
    platform.attach_window(
        imgui.io_mut(),
        &window,
        imgui_winit_support::HiDpiMode::Default,
    );
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
    let mut last_mouse = Vec2::ZERO;

    let clear_color = wgpu::Color {
        r: 0.1,
        g: 0.2,
        b: 0.3,
        a: 1.0,
    };

    event_loop
        .run(|event, elwt| {
            // let _ = (&instance, &adapter, &shader, &pipeline_layout);
            let _ = (&instance, &adapter, &frag_shader, &vert_shader, &pipeline_layout);
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::AboutToWait => window.request_redraw(),
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(new_size) => {
                        surf_cfg.width = new_size.width.max(1);
                        surf_cfg.height = new_size.height.max(1);
                        size = *new_size;
                        surf.configure(&dev, &surf_cfg);
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.key_without_modifiers().as_ref() {
                                Key::Named(NamedKey::Escape) => {
                                    elwt.exit();
                                }
                                
                                Key::Character("w") => {
                                    cam.pos += cam.mov_lin;
                                }
                                Key::Character("s") => {
                                    cam.pos -= cam.mov_lin;
                                }
                                Key::Character("a") => {
                                    cam.pos -= cam.mov_lat;
                                }
                                Key::Character("d") => {
                                    cam.pos += cam.mov_lat;
                                }
                                _ => {},
                            }
                            
                            uniform.cam_pos = cam.pos.extend(0.0).to_array();
                            queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let pos = position.to_logical::<f32>(hidpi_factor);
                        let mouse = Vec2::new(pos.x, pos.y);
                        let mouse_delta = mouse
                            - Vec2::new(size.width as f32, size.height as f32)
                                / (2.0 * hidpi_factor as f32);
                        
                        cam.rotate(mouse_delta);

                        uniform.cam_dir = cam.front.extend(0.0).to_array();
                        uniform.cam_plane_u = cam.plane_u.extend(0.0).to_array();
                        uniform.cam_plane_v = cam.plane_v.extend(0.0).to_array();
                        uniform.mouse = mouse.to_array();
                        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

                        window
                            .set_cursor_position(PhysicalPosition::new(
                                size.width / 2,
                                size.height / 2,
                            ))
                            .unwrap();
                        last_mouse = mouse;
                    }
                    WindowEvent::RedrawRequested => {
                        let delta_time = last_frame.elapsed();
                        let fps = 1.0 / delta_time.as_secs_f32();
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
                                .position([10.0, 10.0], Condition::FirstUseEver)
                                .build(|| {
                                    ui.text(format!("frame_time: {delta_time:?}"));
                                    ui.text(format!("fps: {fps}"));
                                    
                                    let mouse_pos = ui.io().mouse_pos;
                                    ui.text(format!(
                                        "mouse_pos: ({:.1},{:.1})",
                                        mouse_pos[0], mouse_pos[1]
                                    ));
                                    
                                    ui.text(format!("pos: {:?}", cam.pos));
                                });
                        }

                        let mut encoder: wgpu::CommandEncoder =
                            dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        if last_cursor != Some(ui.mouse_cursor()) {
                            last_cursor = Some(ui.mouse_cursor());
                            platform.prepare_render(ui, &window);
                        }

                        {
                            let mut render_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                            render_pass.set_index_buffer(
                                index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
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
        })
        .unwrap();
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
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "error writing log."))
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

    let mut svo = SVO::new(3);
    svo.insert_node(Vec3::from_array([2.0; 3]));
    // svo.gen_random_svo(11482889049544778869);

    info!("filled node count: {}", svo.count_notes());
    svo.log_octree();

    block_on(run(event_loop, window, svo));
}
