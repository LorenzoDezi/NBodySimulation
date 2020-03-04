#version 460 core
out vec4 FragColor;

in vec2 TexCoords;
uniform sampler2D texture_diffuse1;

void main()
{

	vec4 fragColor = texture(texture_diffuse1, TexCoords);
	if (fragColor.a < 0.2) discard;
	else FragColor = fragColor;
}