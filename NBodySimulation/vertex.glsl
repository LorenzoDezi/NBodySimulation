#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec4 aGlobalPosition;
layout(location = 4) in vec4 aAcceleration;
layout(location = 5) in float aRotation;


out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
	TexCoords = aTexCoords;
	//TODO calculate acceleration
	vec3 scaledRotPos = aPos * aGlobalPosition.w;
	scaledRotPos.x = aPos.x * cos(aRotation) + aPos.z * sin(aRotation);
	scaledRotPos.z = -aPos.x * sin(aRotation) + aPos.z * cos(aRotation);
	scaledRotPos.y = aPos.y;
	gl_Position = projection * view * vec4(scaledRotPos + aGlobalPosition.xyz , 1.0f);
}