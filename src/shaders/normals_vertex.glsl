# version 330

in vec3 vertex;
in vec3 normal;

out vec3 _normal;

uniform mat4 projection;
uniform mat3 normals_projection;

void main()
{
    _normal = normals_projection * normal;
    gl_Position = projection * vec4(vertex, 1.0);
}