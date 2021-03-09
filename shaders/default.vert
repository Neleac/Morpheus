#version 330 core
// cast vec2 to vec3 default to 0 for z index
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 texCoord;

out vec2 texel;

uniform mat4 model;
uniform mat4 projection;

void main() {
    gl_Position = projection * model * vec4(Position, 1.0f);
    texel = texCoord;
}
