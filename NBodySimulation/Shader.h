//
// Created by dezio on 07/04/2019.
//

#ifndef SHADER_1_SHADER_H
#define SHADER_1_SHADER_H

#include "glad/glad.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


class Shader {
public:
    unsigned int ID;
	Shader(const GLchar *vertexPath, const GLchar *fragmentPath, const GLchar *geometryPath = nullptr);
    void use();
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec4Float(const std::string &name, float v1, float v2, float v3, float v4) const;
	void setVec3Float(const std::string &name, float v1, float v2, float v3) const;
	void setMat4Float(const std::string &name, const float *mat) const;

private:
    void checkShaderCompilation(unsigned int vertexShader);
    void checkProgramLinking(unsigned int shaderProgram);
    unsigned int compileShader(const char *code, GLenum type);

    std::string extractShaderFromFile(const GLchar *shaderPath) const;
};


#endif //SHADER_1_SHADER_H
