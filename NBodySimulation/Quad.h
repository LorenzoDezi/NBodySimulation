#pragma once

#include "stb_image/stb_image.h"
#include "Shader.h"
#include <iostream>
#include <vector>

class Quad
{
public:
	/*  Functions   */
	Quad(std::string texturePath);
	~Quad();
	int GetVAO();
	void Draw(Shader shader);
	void DrawInstanced(Shader shader, int amount);
private:
	unsigned int VAO;
	unsigned int VBO;
	unsigned int EBO;
	unsigned int texture;
};

