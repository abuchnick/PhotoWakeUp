# version 330

in vec3 _normal;

out vec3 fragColor;

void main()
{
    fragColor = (_normal + 1.0) * 0.5;
}