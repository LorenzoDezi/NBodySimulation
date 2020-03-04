#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glad/glad.h"
#include "cuda_gl_interop.h"
#include "Model.h"
#include "Shader.h"
#include "Mesh.h"
#include "Camera.h"
#include "nBodyKernels.cuh"
#include "cudaUtilities.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include "glfw/glfw3.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include "constants.h"


void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow* w, double xpos, double ypos);
void processInput(GLFWwindow *w);

float lastFrame = 0.0f;
float timeSinceLastStep = 0.0f;
float deltaTime = 0.0f;
bool firstMouse = true;
float lastX = 0.0f;
float lastY = 0.0f;
Camera camera(glm::vec3(0.f, 0.f, 1000.f), glm::vec3(0.f, 1.0f, 0.0f));

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow *window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
	}
	glViewport(0, 0, 800, 600);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouse_callback);
	glEnable(GL_DEPTH_TEST);

	Shader shader("vertex_planet.glsl", "fragment.glsl");
	Shader instancingShader("vertex.glsl", "fragment.glsl");
	
	//Model loading
	Model rockModel("assets/planet.obj");

	unsigned int bufferPositions;
	cudaGraphicsResource *cudaPositionsResource;
	//Positions generation
	float4 * positions, *cudaPositions;
	float4 * cudaVelocities;
	positions = new float4[N];
	size_t size = N * 4 * sizeof(float);
	CHECK(cudaMalloc((void **)&cudaVelocities, size));
	CHECK(cudaMemset(cudaVelocities, 0, size));
	glGenBuffers(1, &bufferPositions);
	glBindBuffer(GL_ARRAY_BUFFER, bufferPositions);
	glBufferData(GL_ARRAY_BUFFER, size, &positions[0], GL_DYNAMIC_DRAW);
	//Mapping the VBO to CUDA
	cudaGraphicsGLRegisterBuffer(&cudaPositionsResource, bufferPositions, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cudaPositionsResource);
	cudaGraphicsResourceGetMappedPointer((void **)&cudaPositions, &size, cudaPositionsResource);
	//Generation with curand
	curandState *devStates;
	int gridSize = (N + BLOCK_DIM) / BLOCK_DIM;
	cudaMalloc((void **)&devStates, N * sizeof(float));
	generatePointInsideSphere << <gridSize, BLOCK_DIM >> > (cudaPositions, devStates);
	CHECK(cudaDeviceSynchronize());
	cudaFree(devStates);

	//Assignment of attributes to VAOs
	std::vector<int> VAOs = rockModel.GetVAOs();
	for (unsigned int i = 0; i < VAOs.size(); i++)
	{
		unsigned int VAO = VAOs[i];
		glBindVertexArray(VAO);
		GLsizei f4size = 4 * sizeof(float);
		glBindBuffer(GL_ARRAY_BUFFER, bufferPositions);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, f4size, (void*)0);
		glVertexAttribDivisor(3, 1);

		glBindVertexArray(0);
	}
	
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	//Preferring L1 over shared in the naive update function (no use of shared)
	CHECK(cudaFuncSetCacheConfig(updateSimple, cudaFuncCache::cudaFuncCachePreferL1));
	while (!glfwWindowShouldClose(window)) {
		//Delta-time per frame logic
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		timeSinceLastStep += deltaTime;
		lastFrame = currentFrame;
		processInput(window);

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//camera transformations setup
		glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.f / 720.f, 0.1f, 10000.0f);
		glm::mat4 view = camera.GetViewMatrix();
		//Shader camera transformation pass
		shader.use();
		shader.setMat4Float("projection", glm::value_ptr(projection));
		shader.setMat4Float("view", glm::value_ptr(view));
		instancingShader.use();
		instancingShader.setFloat("deltaTime", deltaTime);
		instancingShader.setMat4Float("projection", glm::value_ptr(projection));
		instancingShader.setMat4Float("view", glm::value_ptr(view));
		//Physics simulation is updated every TIME_STEP
		if (timeSinceLastStep >= TIME_STEP) {
			updateSimple << <gridSize, BLOCK_DIM >> > (cudaPositions, cudaVelocities);
			CHECK(cudaDeviceSynchronize());
			timeSinceLastStep = 0.0f;
		}		
		//draw objects
		instancingShader.use();
		rockModel.DrawInstanced(instancingShader, N);

		glfwSwapBuffers(window);
		glfwPollEvents();
		
	}
	//TODO: Clear data
	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow * window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow * w, double xpos, double ypos) {
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;
	camera.ProcessMouseMovement(xoffset, yoffset);
}

void processInput(GLFWwindow *w) {
	if (glfwGetKey(w, GLFW_KEY_ESCAPE)) {
		glfwSetWindowShouldClose(w, true);
	}
	if (glfwGetKey(w, GLFW_KEY_UP)) {
		camera.ProcessKeyboard(FORWARD, deltaTime);
	}
	if (glfwGetKey(w, GLFW_KEY_DOWN)) {
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	}
	if (glfwGetKey(w, GLFW_KEY_LEFT)) {
		camera.ProcessKeyboard(LEFT, deltaTime);
	}
	if (glfwGetKey(w, GLFW_KEY_RIGHT)) {
		camera.ProcessKeyboard(RIGHT, deltaTime);
	}
}
