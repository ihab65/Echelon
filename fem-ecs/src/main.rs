use bevy::prelude::*;
use gui::GuiPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GuiPlugin)
        .run();
}