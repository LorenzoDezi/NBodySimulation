#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec4 aGlobalPosition;

out vec2 TexCoords;
uniform float deltaTime;
uniform mat4 projection;
uniform mat4 view;

void main()
{
	TexCoords = aTexCoords;
	vec3 scaledLocalPos = aPos * aGlobalPosition.w;
	gl_Position = projection * view * vec4(scaledLocalPos + aGlobalPosition.xyz, 1.0f);
}