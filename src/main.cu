#include "MAC.cuh"
#include "glprocess.cuh"
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLES3/gl3.h>

#define P(x) std::cout<<#x<<" = "<<(x)<<std::endl

const int pperedge = 30;
const int particleNum = pperedge*pperedge*pperedge;

MAC& getMAC(){
	static MAC mac(50, 50, 50, 0.95);
	return mac;
}

Particles& getParticles(){
	static Particles p(particleNum);
	return p;
}

void waddle(float t, float dt, MAC& mac);

int main(int argc, char *argv[])
{
	getParticles().setParticlesUniform(Particles::flflin(1.5, getMAC().Nx()/2.0+0.1, pperedge),
			Particles::flflin(1.5, getMAC().Ny()/2.0+0.1, pperedge),
			Particles::flflin(getMAC().Nz()/2.0+0.1, getMAC().Nz()-2.5, pperedge));

//	for (int i = 0; i < getMAC().Nx(); ++i) {
//		for (int j = 0; j < getMAC().Ny(); ++j) {
//			for (int k = 0; k < getMAC().Nz(); ++k) {
//				if(i+j+k<=getMAC().Nx()/2)
//				getMAC().blockType.p()[k*getMAC().Nx()*getMAC().Ny()+j*getMAC().Nx()+i]=gridType::Solid;
//			}
//		}
//	}

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

	glClearColor(0.4, 0.0, 0.7, 0.95);
	glEnable(GL_DEPTH_TEST);
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(2, VAO);
	glBindVertexArray(VAO[0]);
	createVBO(VBO, &cudaVboRes, cudaGraphicsMapFlagsWriteDiscard);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glBindVertexArray(VAO[1]);
	glGenBuffers(1, VBO+1);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	float Lx(getMAC().Nx()-1);
	float Ly(getMAC().Ny()-1);
	float Lz(getMAC().Nz()-1);
	float vertices[6*2*3*3]={
		1., 1., 1., Lx, 1., 1., 1., Ly, 1.,
		Lx, 1., 1., Lx, Ly, 1., 1., Ly, 1.,
		1., 1., 1., Lx, 1., 1., 1., 1., Lz,
		Lx, 1., 1., 1., 1., Lz, Lx, 1., Lz,
		1., 1., 1., 1., Ly, 1., 1., 1., Lz,
		1., Ly, 1., 1., 1., Lz, 1., Ly, Lz,
		1., 1., Lz, Lx, 1., Lz, 1., Ly, Lz,
		Lx, 1., Lz, Lx, Ly, Lz, 1., Ly, Lz,
		1., Ly, 1., Lx, Ly, 1., 1., Ly, Lz,
		Lx, Ly, 1., 1., Ly, Lz, Lx, Ly, Lz,
		Lx, 1., 1., Lx, Ly, 1., Lx, 1., Lz,
		Lx, Ly, 1., Lx, 1., Lz, Lx, Ly, Lz
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), nullptr);
	glEnableVertexAttribArray(1);

	std::ifstream vertex_shader_file("glsl/vertex_shader.glsl");
	std::stringstream strStream;
	strStream << vertex_shader_file.rdbuf();
	std::string vertex_shader_code = strStream.str();

	std::ifstream fragment_shader_file("glsl/fragment_shader.glsl");
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
	forceFunc = nullptr;
	glutMainLoop();
}

void waddle(float t, float dt, MAC& mac){
	float sign=(int(t/10.0)%2)?1.0:-1.0;
	mac.applyForce(make_float3(0.0, 0.0, sign*9.8), dt);
}
