#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glad/glad.h"
#include "cuda_gl_interop.h"
#include "Model.h"
#include "Shader.h"
#include "Mesh.h"
#include "Camera.h"
#include "nBodyKernels.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include "glfw/glfw3.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>

#define N 1000
#define RADIUS 100
#define MASS_SEED 100


void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow* w, double xpos, double ypos);
void processInput(GLFWwindow *w);

float lastFrame = 0.0f;
float deltaTime = 0.0f;
bool firstMouse = true;
float lastX = 0.0f;
float lastY = 0.0f;
Camera camera(glm::vec3(0.f, 0.f, 5.f), glm::vec3(0.f, 1.0f, 0.0f));

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
	Model rockModel("assets/rock.obj");

	// generate a large list of semi-random model transformation matrices
	glm::mat4* modelMatrices;
	glm::vec4 * positions;
	positions = new glm::vec4[N];
	glm::vec4 * accelerations;
	accelerations = new glm::vec4[N];
	float * rotations;
	rotations = new float[N];
	srand(glfwGetTime()); // initialize random seed	
	for (unsigned int i = 0; i < N; i++)
	{
		//Start acceleration is 0
		accelerations[i] = glm::vec4(0.0f, 0.5f, 0.0f, 1.0f);
		
		//Position: random point inside a sphere of radius RADIUS
		float x, y, z;
		float radius = -RADIUS + (rand() % RADIUS); //random radius
		float theta = rand() % 360; //random angle on xz plane
		float gamma = rand() % 360; //random angle on yz/yx plane
		//Random point inside sphere
		x = radius * cos(theta) * cos(gamma);
		y = radius * sin(gamma);
		z = radius * sin(theta) * cos(gamma);
		// Scale: scale depending on mass
		float mass = (rand() % MASS_SEED) / 100.0f + 0.5f;
		positions[i] = glm::vec4(x, y, z, mass);
		// Rotation: add random rotation around a randomly picked rotation axis vector
		float rotAngle = (rand() % 360);
		rotations[i] = rotAngle;
	}

	// vertex Buffer Objects
	unsigned int buffer_positions, buffer_accelerations, buffer_rotations;
	//CUDA Graphics resources
	cudaGraphicsResource *cudaPositionsResource, *cudaAccelerationsResource;
	//CUDA buffer pointers
	float4 *cudaPositions, *cudaAccelerations;
	//Positions VBO
	size_t size = N * 4 * sizeof(float);
	glGenBuffers(1, &buffer_positions);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_positions);
	glBufferData(GL_ARRAY_BUFFER, size, &positions[0], GL_DYNAMIC_DRAW);
	//Mapping the VBO to CUDA
	cudaGraphicsGLRegisterBuffer(&cudaPositionsResource, buffer_positions, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cudaPositionsResource);
	cudaGraphicsResourceGetMappedPointer((void **)&cudaPositions, &size, cudaPositionsResource);

	//Acceleration VBO
	glGenBuffers(1, &buffer_accelerations);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_accelerations);
	glBufferData(GL_ARRAY_BUFFER, size, &accelerations[0], GL_DYNAMIC_DRAW);
	//Mapping the VBO to CUDA
	cudaGraphicsGLRegisterBuffer(&cudaAccelerationsResource, buffer_accelerations, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cudaAccelerationsResource);
	cudaGraphicsResourceGetMappedPointer((void **)&cudaAccelerations, &size, cudaAccelerationsResource);

	glGenBuffers(1, &buffer_rotations);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_rotations);
	glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), &rotations[0], GL_DYNAMIC_DRAW);

	//TODO Bind also to cuda. Consider passing only positions, and then to the model matrix calculation inside 
	//the shaders. You need position and acceleration, then each shader will calculate the model matrix based on that
	std::vector<int> VAOs = rockModel.GetVAOs();
	for (unsigned int i = 0; i < VAOs.size(); i++)
	{
		unsigned int VAO = VAOs[i];
		glBindVertexArray(VAO);
		// vertex Attributes
		GLsizei f4size = sizeof(glm::vec4);
		glBindBuffer(GL_ARRAY_BUFFER, buffer_positions);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, f4size, (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, buffer_accelerations);
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, f4size, (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, buffer_rotations);
		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);

		glVertexAttribDivisor(3, 1);
		glVertexAttribDivisor(4, 1);
		glVertexAttribDivisor(5, 1);

		glBindVertexArray(0);
	}
	
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	while (!glfwWindowShouldClose(window)) {
		//Delta-time per frame logic
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		processInput(window);

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//camera transformations setup
		glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.f / 720.f, 0.1f, 1000.0f);
		glm::mat4 view = camera.GetViewMatrix();
		//Shader camera transformation pass
		shader.use();
		shader.setMat4Float("projection", glm::value_ptr(projection));
		shader.setMat4Float("view", glm::value_ptr(view));
		instancingShader.use();
		instancingShader.setFloat("deltaTime", deltaTime);
		instancingShader.setMat4Float("projection", glm::value_ptr(projection));
		instancingShader.setMat4Float("view", glm::value_ptr(view));
		updateSimple<<<1, 10>>>(cudaAccelerations, cudaPositions);
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
