# version 330

in vec3 vertex;
in float weight;
out float _weight;

uniform mat4 projection;

void main()
{
    _weight = weight;
    gl_Position = projection * vec4(vertex, 1.0);
}