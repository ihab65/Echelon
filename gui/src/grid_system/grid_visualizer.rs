use bevy::{color::palettes::css::{DARK_GRAY}, prelude::*};
use bevy_egui::{egui, EguiContexts};
use fem_core::grid_system::Grid;

pub struct GridVisualizerPlugin;

impl Plugin for GridVisualizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostStartup, draw_2d_grid)
           .add_systems(PostStartup, draw_3d_grid);
    }
}

// Resource to store viewport rects for interaction
#[derive(Resource, Default)]
pub struct ViewportRects {
    pub rect_2d: Option<egui::Rect>,
    pub rect_3d: Option<egui::Rect>,
}

// ---------------------
// 2D Grid Renderer
// ---------------------
fn draw_2d_grid(
    viewport_rects: Res<ViewportRects>,
    grid: Res<Grid>,
    mut ctx: EguiContexts,
) {
    if let Some(rect) = viewport_rects.rect_2d {
        if let Ok(ctx) = ctx.ctx_mut() {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("grid2d")));
            let spacing = grid.spacing_x; // example
            let width = rect.width();
            let height = rect.height();

            // draw vertical lines
            for i in 0..=((width / spacing) as usize) {
                let x = rect.left() + i as f32 * spacing;
                painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                                     egui::Stroke::new(1.0, egui::Color32::LIGHT_GRAY));
            }

            // draw horizontal lines
            for i in 0..=((height / spacing) as usize) {
                let y = rect.top() + i as f32 * spacing;
                painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                                     egui::Stroke::new(1.0, egui::Color32::LIGHT_GRAY));
            }
        }
    }
}
// ---------------------
// 3D Grid Renderer
// ---------------------
fn draw_3d_grid(grid: Res<Grid>, mut gizmos: Gizmos) {
    for floor in &grid.floors {
        for i in 0..=20 {
            let x = grid.origin.x + i as f32 * grid.spacing_x;
            gizmos.line(
                Vec3::new(x, floor.level, grid.origin.z),
                Vec3::new(x, floor.level, grid.origin.z + grid.spacing_y * 20.0),
                DARK_GRAY,
            );
        }

        for j in 0..=20 {
            let z = grid.origin.z + j as f32 * grid.spacing_y;
            gizmos.line(
                Vec3::new(grid.origin.x, floor.level, z),
                Vec3::new(grid.origin.x + grid.spacing_x * 20.0, floor.level, z),
                DARK_GRAY,
            );
        }
    }
}