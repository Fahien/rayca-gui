// Copyright © 2021-2024
// Author: Antonio Caggiano <info@antoniocaggiano.eu>
// SPDX-License-Identifier: MIT

use std::rc::Rc;

use memoffset::offset_of;
use rayca_core::*;
use rayca_pipe::pipewriter;

pipewriter!(Gui, "shaders/gui.vert.slang", "shaders/gui.frag.slang");

#[repr(transparent)]
pub struct EguiVertex {
    _vertex: egui::epaint::Vertex,
}

impl VertexInput for EguiVertex {
    fn get_bindings() -> Vec<vk::VertexInputBindingDescription> {
        vec![
            vk::VertexInputBindingDescription::default().stride(std::mem::size_of::<Self>() as u32),
        ]
    }

    fn get_attributes() -> Vec<vk::VertexInputAttributeDescription> {
        let pos = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(egui::epaint::Vertex, pos) as u32);

        let uv = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(egui::epaint::Vertex, uv) as u32);

        let col = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(offset_of!(egui::epaint::Vertex, color) as u32);

        vec![pos, uv, col]
    }

    fn get_depth_state<'a>() -> vk::PipelineDepthStencilStateCreateInfo<'a> {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
    }

    fn get_color_blend() -> Vec<vk::PipelineColorBlendAttachmentState> {
        vec![
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
        ]
    }

    fn get_subpass() -> u32 {
        1
    }
}

impl RenderPipeline for PipelineGui {
    fn render(
        &self,
        _frame: &mut Frame,
        _model: Option<&RenderModel>,
        _camera_nodes: &[Handle<Node>],
        _nodes: &[Handle<Node>],
    ) {
    }
}

pub struct GuiTexture {
    staging: Buffer,
    texture: RenderTexture,
    _sampler: RenderSampler,
    _view: ImageView,
    image: RenderImage,
}

pub struct GuiFont {
    /// Keeps a texture for each in-flight frame
    textures: Vec<GuiTexture>,
}

impl GuiTexture {
    pub fn new(
        allocator: &Rc<vk_mem::Allocator>,
        command_buffer: &CommandBuffer,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Self {
        let image = RenderImage::sampled(allocator, width, height, vk::Format::R8G8B8A8_UNORM);
        let view = ImageView::new(&command_buffer.device, &image);
        let sampler = RenderSampler::new(&command_buffer.device);

        let texture = RenderTexture::new(&view, &sampler);

        let staging = Buffer::new_with_size(
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            data.len() as u64,
        );

        let mut ret = Self {
            staging,
            texture,
            _sampler: sampler,
            _view: view,
            image,
        };
        ret.upload(command_buffer, data);
        ret
    }

    pub fn upload(&mut self, command_buffer: &CommandBuffer, data: &[u8]) {
        self.staging.upload_arr(data);
        self.image.copy_from(&self.staging, command_buffer);
    }
}

impl GuiFont {
    pub fn new(
        allocator: &Rc<vk_mem::Allocator>,
        frame: &mut Frame,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Self {
        let mut textures = Vec::new();
        for _ in 0..frame.in_flight_count {
            textures.push(GuiTexture::new(
                allocator,
                &frame.cache.command_buffer,
                data,
                width,
                height,
            ));
        }
        Self { textures }
    }
}

pub struct Gui {
    /// Initialized after the first egui update
    font: Option<GuiFont>,
    /// Primitives might change every frame, hence we keep multiple collection
    primitives: Vec<Vec<RenderPrimitive>>,
    ctx: egui::Context,
    pipeline: PipelineGui,
    allocator: Rc<vk_mem::Allocator>,
}

impl Gui {
    pub fn new(
        #[cfg(target_os = "android")] android_app: &AndroidApp,
        in_flight_frames: usize,
        allocator: &Rc<vk_mem::Allocator>,
        pass: &Pass,
    ) -> Self {
        let pipeline = PipelineGui::new::<EguiVertex>(
            #[cfg(target_os = "android")]
            android_app,
            pass,
        );

        let ctx = egui::Context::default();
        {
            let mut style = ctx.style().as_ref().clone();
            for (_, font) in style.text_styles.iter_mut() {
                font.size = 32.0;
            }
            ctx.set_style(style);
        }

        let mut primitives = Vec::new();
        for _ in 0..in_flight_frames {
            primitives.push(Vec::new())
        }

        Self {
            font: None,
            primitives,
            ctx,
            pipeline,
            allocator: allocator.clone(),
        }
    }

    fn egui(&self, delta: f32, input: &Input, size: Size2) -> egui::FullOutput {
        let mut raw_input = egui::RawInput::default();
        raw_input.predicted_dt = delta;
        if input.w.just_updated() {
            raw_input.events.push(egui::Event::Key {
                key: egui::Key::W,
                physical_key: None,
                pressed: input.w.is_down(),
                repeat: false,
                modifiers: Default::default(),
            })
        }
        if input.s.just_updated() {
            raw_input.events.push(egui::Event::Key {
                key: egui::Key::S,
                physical_key: None,
                pressed: input.s.is_down(),
                repeat: false,
                modifiers: Default::default(),
            })
        }
        let mouse_pos = egui::Pos2 {
            x: input.mouse.position.x,
            y: input.mouse.position.y,
        };
        if input.mouse.just_moved {
            raw_input.events.push(egui::Event::PointerMoved(mouse_pos));
        }
        if input.mouse.left.just_updated() {
            raw_input.events.push(egui::Event::PointerButton {
                pos: mouse_pos,
                button: egui::PointerButton::Primary,
                pressed: input.mouse.left.is_down(),
                modifiers: Default::default(),
            });
        }
        if input.mouse.right.just_updated() {
            raw_input.events.push(egui::Event::PointerButton {
                pos: mouse_pos,
                button: egui::PointerButton::Secondary,
                pressed: input.mouse.right.is_down(),
                modifiers: Default::default(),
            });
        }

        raw_input.screen_rect.replace(egui::Rect::from_min_size(
            Default::default(),
            egui::Vec2::new(size.width as f32, size.height as f32),
        ));

        self.ctx.begin_pass(raw_input);

        egui::Window::new("Title")
            .auto_sized()
            .collapsible(false)
            .show(&self.ctx, |ui| ui.label("Text"));

        self.ctx.end_pass()
    }

    fn update_textures(&mut self, frame: &mut Frame, textures_delta: &egui::TexturesDelta) {
        for (_, image_delta) in &textures_delta.set {
            // Extract pixel data from egui
            let data: Vec<u8> = match &image_delta.image {
                egui::ImageData::Font(image) => image
                    .srgba_pixels(Some(1.0))
                    .flat_map(|color| color.to_array())
                    .collect(),
                _ => unimplemented!(),
            };

            if let Some(font) = self.font.as_mut() {
                font.textures[frame.id].upload(&frame.cache.command_buffer, &data);
            } else {
                self.font.replace(GuiFont::new(
                    &self.allocator,
                    frame,
                    &data,
                    image_delta.image.width() as u32,
                    image_delta.image.height() as u32,
                ));
            }
        }
    }

    fn update_primitives(&mut self, frame_id: usize, shapes: Vec<egui::epaint::ClippedShape>) {
        let clipped_meshes = self.ctx.tessellate(shapes, 1.0);

        // Collect primitives
        for i in 0..clipped_meshes.len() {
            let clipped_mesh = &clipped_meshes[i];
            let egui::epaint::Primitive::Mesh(mesh) = &clipped_mesh.primitive else {
                continue;
            };

            // Update primitive
            let primitives = &mut self.primitives[frame_id];
            if i >= primitives.len() {
                let primitive = RenderPrimitive::empty::<egui::epaint::Vertex>(&self.allocator);

                primitives.push(primitive);
            }
            primitives[i].vertices.upload_arr(&mesh.vertices);
            primitives[i].set_indices(mesh.indices.as_bytes(), vk::IndexType::UINT32);
        }
    }

    /// This might update textures and primitives, so it should be
    /// called before beginning the render pass
    pub fn update(&mut self, delta: f32, input: &Input, frame: &mut Frame) {
        let egui::FullOutput {
            shapes,
            textures_delta,
            ..
        } = self.egui(delta, input, frame.get_size());

        self.update_textures(frame, &textures_delta);
        self.update_primitives(frame.id, shapes);
    }

    /// To be called after `self.update()`
    pub fn draw(&self, frame: &mut Frame) {
        // Draw with Vulkan pipeline
        self.pipeline.bind(&frame.cache);

        frame.set_viewport_and_scissor(1.0);

        let screen_size: Vec2 = frame.get_size().into();
        self.pipeline
            .push_screen_size(&frame.cache.command_buffer, &screen_size);
        let key = DescriptorKey::builder()
            .layout(self.pipeline.get_layout())
            .build();
        self.pipeline.bind_font_image(
            &frame.cache.command_buffer,
            &mut frame.cache.descriptors,
            key,
            &self.font.as_ref().unwrap().textures[frame.id].texture,
        );
        for primitive in &self.primitives[frame.id] {
            self.pipeline.draw(&frame.cache, primitive);
        }
    }
}
