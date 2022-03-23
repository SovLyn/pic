#include "MAC.cuh"
#include "glprocess.cuh"
#include <GLES3/gl3.h>
#define P(x) std::cout<<#x<<" = "<<(x)<<std::endl

int main(int argc, char *argv[])
{
	MAC mac(10, 10, 10);
	getParticles().setParticlesUniform(Particles::flflin(0.1, 0.9, 5), 
			Particles::flflin(0.1, 0.9, 7), 
			Particles::flflin(0.1, 0.9, 9));
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("cuda & OpenGL");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboardOP);
	glutMouseFunc(mouseOP);
	glutMotionFunc(motionOP);
	glutTimerFunc(refreshDelay, timerEvent, 0);
	glutCloseFunc(clean);

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	glClearColor(0.4, 0.0, 0.7, 1.0);
	glEnable(GL_DEPTH_TEST);
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, VAO);
	glBindVertexArray(VAO[0]);
	createVBO(VBO, &cudaVboRes, cudaGraphicsMapFlagsWriteDiscard);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	std::ifstream vertex_shader_file("vertex_shader.glsl");
	std::stringstream strStream;
	strStream << vertex_shader_file.rdbuf();
	std::string vertex_shader_code = strStream.str();

	std::ifstream fragment_shader_file("fragment_shader.glsl");
	strStream.str(std::string());
	strStream<<fragment_shader_file.rdbuf();
	std::string fragment_shader_code = strStream.str();

	unsigned int shaderID = create_shader(vertex_shader_code, fragment_shader_code);
	glUseProgram(shaderID);

	projection_loc = glGetUniformLocation(shaderID, "projection");
	view_loc = glGetUniformLocation(shaderID, "view");
	glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm::value_ptr(projection));
	glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view));

	startTime = std::chrono::high_resolution_clock::now();
	glutMainLoop();
}
