#pragma once
//header guard
#ifndef GLBASE_H
#define GLBASE_H


	class glWrapper : RAIIscope{
	public:
			SDL_Window *window;
			int screenWidth = 1024;
			int screenHeight = 512;
			SDL_GLContext SDLcontext;
			GLuint programID;
			GLuint programRenderID;
			GLuint vertexbuffer;
			GLuint vertexBuffer3d;
			GLuint UVbuffer;
	public: glWrapper();
			int initContext(cl_platform_id platform, cl_device_id* device);
			int initGl();

			cl_mem createGLtexture(size_t width, size_t height, cl_mem_flags flags, GLuint* outTextID);
			cl_mem createGLbuffer(size_t width, size_t height, cl_mem_flags flags);

			int releaseTexture(cl_mem imageCL);
			int acquireTexture(cl_mem imageCL);

			int glOps(image* img);
			int glRender(cl_mem* buffer);
			void glWrapper::loopRender();
			int Cleanup();
	};

#endif