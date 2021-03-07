#version 330 core
out vec4 FragColor;

in vec2 texel;

// to use color, set texture to white
// to use texture, set color to white
uniform vec4 color;
uniform sampler2D tex;

void main() {
    FragColor = color * texture(tex, texel);
}
