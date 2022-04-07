# version 330

in vec2 uv;

out vec3 fragColor;

uniform sampler2D Texture;

void main()
{
    fragColor = texture(Texture, uv).xyz;
}