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
	mat3 particleRotMat = transpose(mat3(view));
	mat4 model = mat4(vec4(particleRotMat[0], 0), vec4(particleRotMat[1], 0), vec4(particleRotMat[2], 0), vec4(aGlobalPosition.xyz, 1.0f));
	gl_Position = projection * view * model * vec4(scaledLocalPos, 1.0);
}