# version 330

in vec3 vertex;
in vec2 uv_coordinates;

out vec2 uv;

uniform mat4 projection;
uniform mat3 joint_rotations[22];

int kintree[22] = int[22]( -1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19 );

void main()
{
    uv = uv_coordinates;
    gl_Position = projection * vec4(vertex, 1.0);
}