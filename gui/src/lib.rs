pub mod grid_system;

use crate::grid_system::{GridVisualizerPlugin, ViewportRects};
use bevy::prelude::*;
use bevy::pbr::PointLight;
use bevy_egui::{EguiPlugin, EguiContexts, egui, EguiPrimaryContextPass};
use fem_core::grid_system::Grid;

pub struct GuiPlugin;

impl Plugin for GuiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
           .add_plugins(GridVisualizerPlugin)
           .add_systems(Startup, setup)
           .add_systems(EguiPrimaryContextPass, gui_system);
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera3d::default());
    commands.spawn(PointLight::default());

    // Insert a default Grid resource
    let mut grid = Grid::new(bevy::math::Vec3::ZERO, 1.0, 1.0, 3.0); // origin, spacing_x, spacing_z, floor_height
    grid.add_floor("Level 1", 0.0);
    grid.add_floor("Level 2", 3.0);
    commands.insert_resource(grid);
    // Insert ViewportRects
    commands.insert_resource(ViewportRects::default());
}

fn gui_system(mut contexts: EguiContexts, mut viewport_rects: ResMut<ViewportRects>) {
    if let Ok(ctx) = contexts.ctx_mut() {
        // --- Top Bar ---
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Workspace selector
                ui.label("Workspace:");
                ui.selectable_value(&mut "Geometry", "Geometry", "Geometry");
                ui.selectable_value(&mut "BCs", "BCs", "BCs");
                ui.selectable_value(&mut "Materials", "Materials", "Materials");
                ui.selectable_value(&mut "Loads", "Loads", "Loads");
                ui.selectable_value(&mut "Sensors", "Sensors", "Sensors");
                ui.selectable_value(&mut "Figure Creation", "Figure Creation", "Figure Creation");

                ui.separator();

                // Tool shortcuts
                ui.horizontal(|ui| {
                    if ui.button("Node").clicked() {println!("Node tool selected");}
                    if ui.button("Element").clicked() {println!("Element tool selected");}
                    if ui.button("BC").clicked() {println!("BC tool selected");}
                    if ui.button("Load").clicked() {println!("Load tool selected");}
                    if ui.button("Plane").clicked() {println!("Plane tool selected");}
                    if ui.button("Mesh/Wall").clicked() {println!("Mesh/Wall tool selected");}
                });

                ui.separator();

                // Right-aligned controls
                ui.with_layout(egui::Layout::right_to_left(bevy_egui::egui::Align::RIGHT), |ui| {
                    if ui.button("Run").clicked() {}
                    if ui.button("Check").clicked() {}
                    ui.label("Solver: Default");
                });
            });
        });

        // --- Side Panel ---
        egui::SidePanel::right("side_panel").show(ctx, |ui| {
            ui.heading("Properties");
            ui.label("Select an entity to see its metadata...");
            // Here you can dynamically show node coords, material props, BC options, etc.
        });

        // --- Bottom Bar ---
        egui::TopBottomPanel::bottom("bottom_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Status: Ready");
                ui.with_layout(egui::Layout::right_to_left(bevy_egui::egui::Align::RIGHT), |ui| {
                    ui.label("Progress: 0%");
                });
            });
        });

        // --- Viewport ---
        // --- Central Panel: Dual Viewports ---
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // 2D Plan View
                ui.vertical(|ui| {
                    ui.label("2D Plan View");
                    let size_2d = ui.available_size() * 0.5;
                    
                    let (rect_2d, _response) = ui.allocate_exact_size(size_2d, egui::Sense::hover());
                    viewport_rects.rect_2d = Some(rect_2d);
                    
                    ui.painter().rect_stroke(
                        rect_2d, 0.0,
                        egui::Stroke::new(1.0, egui::Color32::LIGHT_GRAY),
                        egui::StrokeKind::Outside
                    );
                });

                // 3D Perspective View
                ui.vertical(|ui| {
                    ui.label("3D Perspective View");
                    let size_3d = ui.available_size() * 0.5;
                    
                    let (rect_3d, _response) = ui.allocate_exact_size(size_3d, egui::Sense::hover());
                    viewport_rects.rect_3d = Some(rect_3d);
                    
                    ui.painter().rect_stroke(
                        rect_3d, 0.0,
                        egui::Stroke::new(1.0, egui::Color32::DARK_GRAY),
                        egui::StrokeKind::Outside
                    );
                });
            });
        });
    }
}