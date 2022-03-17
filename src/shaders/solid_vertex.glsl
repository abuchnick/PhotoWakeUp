# version 330

in vec3 vertex;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex, 1.0);
}