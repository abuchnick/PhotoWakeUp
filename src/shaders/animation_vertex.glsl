# version 330

in vec3 vertex;
in vec2 uv_coordinates;
// in float skinning_weights[22];

out vec2 uv;

uniform mat4 projection;
// uniform mat4 joint_rotations[22];

void main()
{
    uv = uv_coordinates;
    // mat4 T = mat4(0.0);
    // for(int i = 0 ; i < 22 ; ++i)
    // {
    //     T += skinning_weights[i] * joint_rotations[i];
    // }
    gl_Position = projection * vec4(vertex, 1.0);
}