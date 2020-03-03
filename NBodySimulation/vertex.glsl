#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec4 aGlobalPosition;
layout(location = 4) in vec4 aAcceleration;
layout(location = 5) in float aRotation;


out vec2 TexCoords;
uniform float deltaTime;
uniform mat4 projection;
uniform mat4 view;

void main()
{
	TexCoords = aTexCoords;
	vec3 scaledRotPos = aPos * aGlobalPosition.w;
	scaledRotPos.x = aPos.x * cos(aRotation) + aPos.z * sin(aRotation);
	scaledRotPos.z = -aPos.x * sin(aRotation) + aPos.z * cos(aRotation);
	scaledRotPos.y = aPos.y;
	//TODO: Check if it works -> see if it works done by cuda
	vec3 translation = aGlobalPosition.xyz; //+  (1 / 2) * aAcceleration.xyz * deltaTime * deltaTime;
	gl_Position = projection * view * vec4(scaledRotPos + translation , 1.0f);
}