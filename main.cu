#include "MAC.cuh"
#include "glprocess.cuh"

int main(int argc, char *argv[])
{
	MAC mac(10, 10, 10);
	getParticles().setParticlesUniform(Particles::flflin(0.1, 0.6, 5), 
			Particles::flflin(0.4, 0.9, 7), 
			Particles::flflin(0.1, 0.9, 3));
	
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

	glClearColor(0.4, 0.0, 0.7, 1.0);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, windowWidth, windowHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, static_cast<float>(windowWidth)/static_cast<float>(windowHeight), 0.1, 10.0);

	createVBO(&vbo, &cudaVboRes, cudaGraphicsMapFlagsWriteDiscard);
	startTime = std::chrono::high_resolution_clock::now();
	glutMainLoop();
}
