#version 330 core
out vec4 FragColor;

in vec2 texel;

uniform sampler2D textureMap;

void main() {
    vec4 texColor = texture(textureMap, texel);
    if (texColor.a < 0.1)
        discard;

    FragColor = texColor; 
}
